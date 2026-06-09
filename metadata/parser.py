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

    def get_detections(self) -> List[dict]:
        """
        Return the list of per-frame DMS detections.
        Path: inference_data.dms.detections
        """
        try:
            dets = self._data["inference_data"]["dms"]["detections"]
            logger.info(f"Found {len(dets)} detections")
            return dets
        except KeyError:
            logger.error("No detections found at inference_data.dms.detections")
            return []

    def get_events(self) -> List[EventInfo]:
        """
        Return all events from events_data.
        Path: inference_data.events_data.alerts
        """
        try:
            raw_events = self._data["inference_data"]["events_data"]["alerts"]
        except KeyError:
            logger.error("No events_data found in metadata")
            return []

        events = []
        for e in raw_events:
            events.append(
                EventInfo(
                    event_code=e.get("event_code", ""),
                    start_offset=int(e.get("start_timestamp", 0)),
                    end_offset=int(e.get("end_timestamp", 0)),
                    alert_id=e.get("alert_id"),
                    details=e.get("details"),
                    uuid=e.get("uuid"),
                )
            )
        logger.info(f"Found {len(events)} events in metadata")
        return events

    def get_event_by_code(self, event_code: str) -> Optional[EventInfo]:
        """Find the first event matching the given event code."""
        for event in self.get_events():
            if event.event_code == event_code:
                return event
        logger.warning(f"No event found with code {event_code}")
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
