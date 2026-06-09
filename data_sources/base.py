"""
Abstract base class for data sources.

Any data source (MongoDB, PostgreSQL, etc.) must implement this interface.
This makes it trivial to add new data sources in the future.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AlertData:
    """Standardized alert data returned by any data source."""
    alert_id: int
    event_code: str
    device_id: Optional[str] = None
    tenant_id: Optional[int] = None
    vehicle_id: Optional[int] = None
    video_s3_path: Optional[str] = None          # s3:// path to inputVideo.mp4
    metadata_s3_path: Optional[str] = None        # s3:// path to metadata.txt
    video_http_path: Optional[str] = None         # https:// videoPath (from alert collection)
    metadata_http_path: Optional[str] = None      # https:// metadataPath
    start_offset: Optional[int] = None            # event start offset (ms)
    end_offset: Optional[int] = None              # event end offset (ms)
    source: str = "unknown"                       # "video_requests_v2" or "alert"
    raw: dict = field(default_factory=dict)       # original document for debugging


class BaseDataSource(ABC):
    """Abstract interface for fetching alert data."""

    @abstractmethod
    def fetch(self, alert_id: int) -> Optional[AlertData]:
        """Fetch alert data for a given alert_id. Returns None if not found."""
        ...

    @abstractmethod
    def close(self):
        """Clean up connections."""
        ...
