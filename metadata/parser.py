"""
Metadata parser for metadata.txt files.

metadata.txt is a JSON file containing:
  - inference_data.dms.detections: list of per-frame detection dicts (~600 for 1min @ 10fps)
  - inference_data.events_data.alerts: list of events with start/end timestamps and event codes
"""

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from loguru import logger


@dataclass
class EventInfo:
    """Parsed event information from metadata.txt."""
    event_code: str
    start_offset: int      # ms from video start
    end_offset: int        # ms from video start
    alert_id: Optional[int] = None
    details: Optional[str] = None
    uuid: Optional[str] = None


class MetadataParser:
    """Parse metadata.txt (JSON) for detections and event timing."""

    def __init__(self, metadata_path: str):
        self._path = metadata_path
        self._data = None
        self._load()

    def _load(self):
        with open(self._path, "r") as f:
            self._data = json.load(f)
        logger.info(f"Loaded metadata from {self._path}")

    @property
    def raw(self) -> dict:
        """Access the raw metadata dict."""
        return self._data

    def get_detections(self, alert_id: int = None) -> List[dict]:
        """
        Return the list of per-frame DMS detections.
        Path: inference_data.dms.detections
        """
        tag = f"[{alert_id}] " if alert_id else ""
        try:
            dets = self._data["inference_data"]["dms"]["detections"]
            logger.info(f"{tag}Found {len(dets)} detections")
            return dets
        except KeyError:
            logger.error(f"{tag}No detections found at inference_data.dms.detections")
            return []

    def get_events(self, alert_id: int = None) -> List[EventInfo]:
        """
        Return all events from events_data and alerts_data.
        Paths:
          - inference_data.events_data.alerts
          - inference_data.alerts_data.alerts  (fallback / additional)
        Both share the same schema (event_code, start_timestamp, end_timestamp, ...).
        """
        tag = f"[{alert_id}] " if alert_id else ""
        seen_uuids: set = set()
        events: List[EventInfo] = []

        for key in ("events_data", "alerts_data"):
            try:
                raw_events = self._data["inference_data"][key]["alerts"]
            except KeyError:
                continue

            for e in raw_events:
                # Deduplicate by uuid if available
                uuid = e.get("uuid")
                if uuid and uuid in seen_uuids:
                    continue
                if uuid:
                    seen_uuids.add(uuid)

                events.append(
                    EventInfo(
                        event_code=e.get("event_code", ""),
                        start_offset=int(e.get("start_timestamp", 0)),
                        end_offset=int(e.get("end_timestamp", 0)),
                        alert_id=e.get("alert_id"),
                        details=e.get("details"),
                        uuid=uuid,
                    )
                )

        logger.info(f"{tag}Found {len(events)} events in metadata")
        return events

    def get_event_by_code(self, event_code: str, alert_id: int = None) -> Optional[EventInfo]:
        """Find the first event matching the given event code."""
        tag = f"[{alert_id}] " if alert_id else ""
        for event in self.get_events(alert_id=alert_id):
            if event.event_code == event_code:
                return event
        logger.warning(f"{tag}No event found with code {event_code}")
        return None

    def get_detection_keys(self) -> List[str]:
        """Return all available detection field names (for CLI help)."""
        dets = self.get_detections()
        if dets:
            return sorted(dets[0].keys())
        return []

    def get_dms_frame_rate(self) -> int:
        """Return the DMS processing frame rate from metadata."""
        try:
            return int(self._data["inference_data"]["dms"].get("process_fps", 10))
        except (KeyError, TypeError):
            return 10
