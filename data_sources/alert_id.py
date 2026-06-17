"""
Alert ID data source.

Fetches alert data using a multi-strategy fallback chain:
  1. AVC API            - preferred, queries the analytics AVC API for S3 bucket paths
                          (same API used by syncalert.py)
  2. video_requests_v2  - MongoDB collection with direct S3 video paths in retrieved_message
  3. alert              - MongoDB collection with videoPath (HTTPS URL) and metadataPath

The AVC API is tried first. If it fails for an alert, the MongoDB collections
are used as fallback (video_requests_v2 first, then alert).

Event metadata (event_code, start/end offsets) is always backfilled from the
MongoDB alert collection since the AVC API does not return this information.
"""

import re
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

import p_tqdm
import multiprocessing
import pymongo
import requests
from loguru import logger
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry

from ..config import (
    MONGO_DB,
    MONGO_COLLECTION_VIDEO_REQUESTS,
    MONGO_COLLECTION_ALERT,
)
from .base import AlertData, BaseDataSource


# ── AVC API config ──────────────────────────────────────────────────────────
AVC_API_ENDPOINT = "https://analytics-kpis.netradyne.info/avc_api"


def _requests_retry_session(
    retries=5,
    backoff_factor=1,
    status_forcelist=(400, 429, 500, 502, 503, 504),
    session=None,
):
    """Retry strategy for requests (same as syncalert.py)."""
    session = session or requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def _query_avc_api_worker(alert_id: int) -> Optional[Tuple[str, str, str]]:
    """
    Standalone worker function for p_tqdm parallel AVC API calls.
    Queries the AVC API for a single alert_id and returns
    (video_s3, video_http, metadata_s3) or None on failure.

    Must be a top-level function (not a method) so it can be pickled
    by multiprocessing.
    """
    params = {
        "input_data": {"alert_id": alert_id},
        "anonymize_environment": "primary",
        "source": "debug",
        "api_version": "v2",
    }
    try:
        response = _requests_retry_session(retries=5).post(
            AVC_API_ENDPOINT, json=params, timeout=60,
        )
        result = response.json()
        if result.get("msg") != "success":
            logger.warning(
                f"[{alert_id}] AVC API returned: {result.get('msg', result)}"
            )
            return None

        # Extract S3 paths from response
        s3_bucket = result.get("s3_bucket")
        if not s3_bucket:
            return None

        if isinstance(s3_bucket, list):
            s3_bucket = s3_bucket[0]

        tokens = s3_bucket.split("/")
        bucket = tokens[0]
        prefix = "/".join(tokens[1:]).rstrip("/")

        video_s3 = f"s3://{bucket}/{prefix}/8.mp4"
        video_http = f"https://{bucket}.s3.amazonaws.com/{prefix}/8.mp4"
        metadata_s3 = f"s3://{bucket}/{prefix}/metadata.txt"

        logger.info(f"[{alert_id}] AVC API success -> {video_s3}")
        return video_s3, video_http, metadata_s3

    except Exception as e:
        logger.warning(f"[{alert_id}] AVC API error: {e}")
        return None


class AlertIdSource(BaseDataSource):
    """
    Fetch alert data using AVC API (primary) with MongoDB fallback.

    Strategy chain:
      1. AVC API  ->  S3 bucket path  (+ alert collection for event metadata)
      2. video_requests_v2  ->  direct S3 paths  (+ alert collection for offsets)
      3. alert collection  ->  HTTP URLs converted to S3 paths
    """

    def __init__(self, uri: str, db_name: str = MONGO_DB):
        self._client = pymongo.MongoClient(uri)
        self._db = self._client[db_name]
        self._vr_col = self._db[MONGO_COLLECTION_VIDEO_REQUESTS]
        self._alert_col = self._db[MONGO_COLLECTION_ALERT]
        logger.info(f"Connected to MongoDB: {db_name}")

    def fetch(self, alert_id: int) -> Optional[AlertData]:
        """
        Try AVC API first, then video_requests_v2,
        then fall back to alert collection.
        """
        # Strategy 1: AVC API
        data = self._fetch_from_avc_api(alert_id)
        if data is not None:
            return data

        # Strategy 2: video_requests_v2
        data = self._fetch_from_video_requests(alert_id)
        if data is not None:
            self._backfill_offsets_single(data)
            return data

        # Strategy 3: alert collection
        data = self._fetch_from_alert(alert_id)
        if data is not None:
            return data

        logger.warning(f"Alert {alert_id} not found in any source")
        return None

    def fetch_batch(self, alert_ids: List[int]) -> Dict[int, AlertData]:
        """
        Fetch all alerts using a multi-strategy approach:
          1. AVC API (per alert)
          2. video_requests_v2 (batch query)
          3. alert collection (batch query)
        Returns a dict of {alert_id: AlertData} for found alerts.
        """
        results: Dict[int, AlertData] = {}
        remaining = set(alert_ids)

        # ── Strategy 1: AVC API (parallel via p_tqdm) ────────────────────────
        logger.info(f"Trying AVC API for {len(remaining)} alert(s) in parallel...")
        avc_hits: Dict[int, Tuple[str, str, str]] = {}  # aid -> (video_s3, video_http, metadata_s3)

        avc_input_list = list(remaining)
        avc_results = p_tqdm.p_map(
            _query_avc_api_worker,
            avc_input_list,
            desc="AVC API",
            num_cpus=multiprocessing.cpu_count(),
        )

        for aid, s3_info in zip(avc_input_list, avc_results):
            if s3_info is not None:
                avc_hits[aid] = s3_info

        remaining -= set(avc_hits.keys())

        # Batch backfill event metadata from alert collection for AVC API hits
        if avc_hits:
            logger.info(
                f"Found {len(avc_hits)} via AVC API, "
                f"backfilling metadata from alert collection..."
            )
            alert_cursor = self._alert_col.find(
                {"alertId": {"$in": list(avc_hits.keys())}},
                {
                    "alertId": 1, "eventCode": 1,
                    "startOffset": 1, "endOffset": 1,
                    "deviceId": 1, "tenantId": 1,
                },
            )
            alert_metadata = {}
            for doc in alert_cursor:
                alert_metadata[doc.get("alertId")] = doc

            for aid, (video_s3, video_http, metadata_s3) in avc_hits.items():
                doc = alert_metadata.get(aid, {})
                results[aid] = AlertData(
                    alert_id=aid,
                    event_code=doc.get("eventCode", ""),
                    device_id=doc.get("deviceId"),
                    tenant_id=doc.get("tenantId"),
                    video_s3_path=video_s3,
                    video_http_path=video_http,
                    metadata_s3_path=metadata_s3,
                    start_offset=doc.get("startOffset"),
                    end_offset=doc.get("endOffset"),
                    source="avc_api",
                )

        logger.info(
            f"AVC API: found {len(avc_hits)}, "
            f"{len(remaining)} remaining"
        )

        # ── Strategy 2: video_requests_v2 (batch) ───────────────────────────
        if remaining:
            logger.info(f"Querying video_requests_v2 for {len(remaining)} alert(s)...")
            vr_cursor = self._vr_col.find({"alert_id": {"$in": list(remaining)}})
            vr_found = 0
            for doc in tqdm(vr_cursor, total=len(remaining), desc="video_requests_v2"):
                aid = doc.get("alert_id")
                ad = self._parse_video_request_doc(aid, doc)
                if ad is not None:
                    results[aid] = ad
                    remaining.discard(aid)
                    vr_found += 1

            logger.info(
                f"Found {vr_found} in video_requests_v2, "
                f"{len(remaining)} remaining"
            )

            # Backfill offsets from alert collection for video_requests_v2 hits
            vr_ids = [
                aid for aid, ad in results.items()
                if ad.source == "video_requests_v2"
            ]
            if vr_ids:
                logger.info(
                    f"Backfilling offsets from alert collection "
                    f"for {len(vr_ids)} alert(s)..."
                )
                offset_cursor = self._alert_col.find(
                    {"alertId": {"$in": vr_ids}},
                    {"alertId": 1, "startOffset": 1, "endOffset": 1},
                )
                backfilled = 0
                for doc in offset_cursor:
                    aid = doc.get("alertId")
                    if aid in results:
                        so = doc.get("startOffset")
                        eo = doc.get("endOffset")
                        if so is not None and eo is not None:
                            results[aid].start_offset = so
                            results[aid].end_offset = eo
                            backfilled += 1
                logger.info(
                    f"Backfilled offsets for {backfilled}/{len(vr_ids)} alert(s)"
                )

        # ── Strategy 3: alert collection (batch) ────────────────────────────
        if remaining:
            logger.info(f"Querying alert collection for {len(remaining)} alert(s)...")
            alert_cursor = self._alert_col.find(
                {"alertId": {"$in": list(remaining)}}
            )
            found_in_alert = 0
            for doc in tqdm(alert_cursor, total=len(remaining), desc="alert collection"):
                aid = doc.get("alertId")
                ad = self._parse_alert_doc(aid, doc)
                if ad is not None:
                    results[aid] = ad
                    remaining.discard(aid)
                    found_in_alert += 1
            logger.info(f"Found {found_in_alert} in alert collection")

        if remaining:
            logger.warning(
                f"{len(remaining)} alert(s) not found in any source: "
                f"{sorted(remaining)}"
            )

        return results

    # ── AVC API ──────────────────────────────────────────────────────────────

    @staticmethod
    def _query_avc_api(alert_id: int) -> Optional[dict]:
        """
        Call the AVC API with an alert_id to get S3 paths.
        Uses the same API and params as syncalert.py's query_api().

        Returns the API response dict on success, None on failure.
        """
        params = {
            "input_data": {"alert_id": alert_id},
            "anonymize_environment": "primary",
            "source": "debug",
            "api_version": "v2",
        }
        logger.info(f"[{alert_id}] Querying AVC API...")
        try:
            response = _requests_retry_session(retries=5).post(
                AVC_API_ENDPOINT, json=params, timeout=60,
            )
            result = response.json()
            if result.get("msg") == "success":
                logger.info(f"[{alert_id}] AVC API success")
                return result
            else:
                logger.warning(
                    f"[{alert_id}] AVC API returned: {result.get('msg', result)}"
                )
                return None
        except Exception as e:
            logger.warning(f"[{alert_id}] AVC API error: {e}")
            return None

    @staticmethod
    def _extract_s3_from_avc(api_result: dict) -> Optional[Tuple[str, str, str]]:
        """
        Extract video and metadata S3 paths from AVC API response.

        The AVC API returns s3_bucket as "<bucket>/<prefix>".
        We construct:
          - video:    s3://<bucket>/<prefix>/8/inputVideo.mp4
          - metadata: s3://<bucket>/<prefix>/metadata.txt
          - Also a synthetic HTTP URL so that S3Handler.resolve_video_path()
            can extract the session key for its fallback strategies.

        Returns (video_s3_path, video_http_path, metadata_s3_path) or None.
        """
        s3_bucket = api_result.get("s3_bucket")
        if not s3_bucket:
            return None

        if isinstance(s3_bucket, list):
            s3_bucket = s3_bucket[0]

        tokens = s3_bucket.split("/")
        bucket = tokens[0]
        prefix = "/".join(tokens[1:]).rstrip("/")

        video_s3 = f"s3://{bucket}/{prefix}/8.mp4"
        # Synthetic HTTP URL for resolve_video_path() fallback strategies
        video_http = f"https://{bucket}.s3.amazonaws.com/{prefix}/8.mp4"
        metadata_s3 = f"s3://{bucket}/{prefix}/metadata.txt"
        return video_s3, video_http, metadata_s3

    def _query_avc_api_for_s3(
        self, alert_id: int,
    ) -> Optional[Tuple[str, str, str]]:
        """Query AVC API and extract S3 paths.
        Returns (video_s3, video_http, metadata_s3) or None."""
        result = self._query_avc_api(alert_id)
        if result is None:
            return None
        return self._extract_s3_from_avc(result)

    def _fetch_from_avc_api(self, alert_id: int) -> Optional[AlertData]:
        """
        Fetch alert data via AVC API (for S3 paths) + alert collection
        (for event_code, offsets, device/tenant info).
        """
        s3_info = self._query_avc_api_for_s3(alert_id)
        if s3_info is None:
            return None

        video_s3, video_http, metadata_s3 = s3_info

        # Get event metadata from alert collection
        doc = self._alert_col.find_one(
            {"alertId": alert_id},
            {
                "eventCode": 1, "startOffset": 1, "endOffset": 1,
                "deviceId": 1, "tenantId": 1,
            },
        )

        event_code = ""
        start_offset = None
        end_offset = None
        device_id = None
        tenant_id = None

        if doc:
            event_code = doc.get("eventCode", "")
            start_offset = doc.get("startOffset")
            end_offset = doc.get("endOffset")
            device_id = doc.get("deviceId")
            tenant_id = doc.get("tenantId")
        else:
            logger.warning(
                f"[{alert_id}] AVC API succeeded but alert not found in "
                f"MongoDB alert collection (no event metadata)"
            )

        return AlertData(
            alert_id=alert_id,
            event_code=event_code,
            device_id=device_id,
            tenant_id=tenant_id,
            video_s3_path=video_s3,
            video_http_path=video_http,
            metadata_s3_path=metadata_s3,
            start_offset=start_offset,
            end_offset=end_offset,
            source="avc_api",
        )

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

    # ── helpers ──────────────────────────────────────────────────────────────

    def _backfill_offsets_single(self, data: AlertData):
        """Backfill start/end offsets from alert collection for a single AlertData."""
        doc = self._alert_col.find_one(
            {"alertId": data.alert_id},
            {"startOffset": 1, "endOffset": 1},
        )
        if doc:
            so = doc.get("startOffset")
            eo = doc.get("endOffset")
            if so is not None and eo is not None:
                data.start_offset = so
                data.end_offset = eo

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


# Backward compatibility alias
MongoDBSource = AlertIdSource
