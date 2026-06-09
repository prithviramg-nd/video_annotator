"""
MongoDB data source.

Fetches alert data from two MongoDB collections:
  1. video_requests_v2 - preferred, has direct S3 video paths in retrieved_message
  2. alert             - fallback, has videoPath (HTTPS URL) and metadataPath

The video_requests_v2 collection is tried first. If the alert is not found or
the video hasn't been retrieved yet, the alert collection is used as fallback.
"""

import re
from typing import Dict, List, Optional
from urllib.parse import urlparse

import pymongo
from loguru import logger

from ..config import (
    MONGO_URI,
    MONGO_DB,
    MONGO_COLLECTION_VIDEO_REQUESTS,
    MONGO_COLLECTION_ALERT,
)
from .base import AlertData, BaseDataSource


class MongoDBSource(BaseDataSource):
    """Fetch alert data from MongoDB."""

    def __init__(self, uri: str = MONGO_URI, db_name: str = MONGO_DB):
        self._client = pymongo.MongoClient(uri)
        self._db = self._client[db_name]
        self._vr_col = self._db[MONGO_COLLECTION_VIDEO_REQUESTS]
        self._alert_col = self._db[MONGO_COLLECTION_ALERT]
        logger.info(f"Connected to MongoDB: {db_name}")

    def fetch(self, alert_id: int) -> Optional[AlertData]:
        """
        Try video_requests_v2 first (for direct S3 paths),
        fall back to alert collection.
        """
        data = self._fetch_from_video_requests(alert_id)
        if data is None:
            data = self._fetch_from_alert(alert_id)
        if data is None:
            logger.warning(f"Alert {alert_id} not found in any MongoDB collection")
        return data

    def fetch_batch(self, alert_ids: List[int]) -> Dict[int, AlertData]:
        """
        Fetch all alerts in two bulk queries using $in.
        Returns a dict of {alert_id: AlertData} for found alerts.
        """
        results: Dict[int, AlertData] = {}
        remaining = set(alert_ids)

        # ── Batch query: video_requests_v2 ───────────────────────────────────
        logger.info(f"Querying video_requests_v2 for {len(remaining)} alert(s)...")
        vr_cursor = self._vr_col.find({"alert_id": {"$in": list(remaining)}})
        for doc in vr_cursor:
            aid = doc.get("alert_id")
            ad = self._parse_video_request_doc(aid, doc)
            if ad is not None:
                results[aid] = ad
                remaining.discard(aid)

        logger.info(
            f"Found {len(results)} in video_requests_v2, "
            f"{len(remaining)} remaining"
        )

        # ── Batch query: alert collection (for remaining) ────────────────────
        if remaining:
            logger.info(f"Querying alert collection for {len(remaining)} alert(s)...")
            alert_cursor = self._alert_col.find(
                {"alertId": {"$in": list(remaining)}}
            )
            found_in_alert = 0
            for doc in alert_cursor:
                aid = doc.get("alertId")
                ad = self._parse_alert_doc(aid, doc)
                if ad is not None:
                    results[aid] = ad
                    remaining.discard(aid)
                    found_in_alert += 1
            logger.info(f"Found {found_in_alert} in alert collection")

        if remaining:
            logger.warning(
                f"{len(remaining)} alert(s) not found in any collection: "
                f"{sorted(remaining)}"
            )

        return results

    # ── video_requests_v2 ────────────────────────────────────────────────────

    def _fetch_from_video_requests(self, alert_id: int) -> Optional[AlertData]:
        doc = self._vr_col.find_one({"alert_id": alert_id})
        if doc is None:
            logger.debug(f"Alert {alert_id}: not in video_requests_v2")
            return None
        return self._parse_video_request_doc(alert_id, doc)

    def _parse_video_request_doc(self, alert_id: int, doc: dict) -> Optional[AlertData]:
        """Parse a video_requests_v2 document into AlertData."""
        event_code = doc.get("event_code", "")
        device_id = doc.get("device_id")
        tenant_id = doc.get("tenant_id")
        vehicle_id = doc.get("vehicle_id")

        # Extract S3 video and metadata paths from retrieved_message
        video_s3 = None
        metadata_s3 = None
        retrieved = doc.get("retrieved_message", {})
        videos = retrieved.get("videos", {})

        # Find the 8_ camera key (DMS camera)
        for key, info in videos.items():
            if key.startswith("8_"):
                http_video = info.get("s3VideoPath", "")
                http_meta = info.get("metadata", "")
                video_s3 = self._http_to_s3(http_video) if http_video else None
                metadata_s3 = self._http_to_s3(http_meta) if http_meta else None
                break

        if video_s3 is None:
            logger.debug(
                f"Alert {alert_id}: found in video_requests_v2 but no 8_ video "
                f"(retrieved_status={doc.get('retrieved_status')})"
            )
            return None

        return AlertData(
            alert_id=alert_id,
            event_code=event_code,
            device_id=device_id,
            tenant_id=tenant_id,
            vehicle_id=vehicle_id,
            video_s3_path=video_s3,
            metadata_s3_path=metadata_s3,
            source="video_requests_v2",
        )

    # ── alert collection ─────────────────────────────────────────────────────

    def _fetch_from_alert(self, alert_id: int) -> Optional[AlertData]:
        doc = self._alert_col.find_one({"alertId": alert_id})
        if doc is None:
            logger.debug(f"Alert {alert_id}: not in alert collection")
            return None
        return self._parse_alert_doc(alert_id, doc)

    def _parse_alert_doc(self, alert_id: int, doc: dict) -> Optional[AlertData]:
        """Parse an alert collection document into AlertData."""
        event_code = doc.get("eventCode", "")
        video_http = doc.get("videoPath", "")
        metadata_http = doc.get("metadataPath", "")
        start_offset = doc.get("startOffset")
        end_offset = doc.get("endOffset")
        device_id = doc.get("deviceId")
        tenant_id = doc.get("tenantId")
        vehicle_id = doc.get("residue", {}).get("vehicle_id")

        # Convert HTTP paths to S3 paths
        video_s3 = self._resolve_video_s3(video_http) if video_http else None
        metadata_s3 = self._http_to_s3(metadata_http) if metadata_http else None

        return AlertData(
            alert_id=alert_id,
            event_code=event_code,
            device_id=device_id,
            tenant_id=tenant_id,
            vehicle_id=int(vehicle_id) if vehicle_id else None,
            video_s3_path=video_s3,
            video_http_path=video_http,
            metadata_s3_path=metadata_s3,
            metadata_http_path=metadata_http,
            start_offset=start_offset,
            end_offset=end_offset,
            source="alert",
        )

    # ── URL helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _http_to_s3(http_url: str) -> Optional[str]:
        """Convert an HTTPS S3 URL to an s3:// path."""
        if not http_url:
            return None
        parsed = urlparse(http_url)
        bucket = parsed.hostname.split(".")[0]
        key = parsed.path.lstrip("/")
        return f"s3://{bucket}/{key}"

    @staticmethod
    def _resolve_video_s3(video_http: str) -> Optional[str]:
        """
        From the alert collection's videoPath (which may point to trimmedVideos
        or outputVideo.mp4), resolve to the 8/inputVideo.mp4 path.

        The actual video download will try multiple strategies in the S3 handler,
        so here we just extract the session key and construct the primary path.
        """
        if not video_http:
            return None
        parsed = urlparse(video_http)
        bucket = parsed.hostname.split(".")[0]
        key = parsed.path.lstrip("/")

        # Extract session key: everything before /8/ or /trimmedVideos/ or /outputVideo
        # e.g. "N406.../ce3bbd82.../trimmedVideos/..." -> "N406.../ce3bbd82.../"
        cam_match = re.search(r"^(.*?)/(?:\d+/|trimmedVideos/|outputVideo)", key)
        if cam_match:
            session_key = cam_match.group(1) + "/"
        else:
            session_key = key[: key.rfind("/") + 1]

        return f"s3://{bucket}/{session_key}8/inputVideo.mp4"

    def close(self):
        self._client.close()
        logger.info("MongoDB connection closed")
