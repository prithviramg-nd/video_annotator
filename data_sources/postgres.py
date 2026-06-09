"""
PostgreSQL data source (stub for future use).

Will be used when alert data needs to be fetched by AVID or other
identifiers not available in MongoDB.
"""

from typing import Optional

from loguru import logger

from .base import AlertData, BaseDataSource


class PostgresSource(BaseDataSource):
    """Fetch alert data from PostgreSQL. (Future implementation)"""

    def __init__(self, connection_string: str):
        self._conn_str = connection_string
        # TODO: establish connection using psycopg2
        logger.info("PostgresSource initialized (stub)")

    def fetch(self, alert_id: int) -> Optional[AlertData]:
        """Fetch alert data by alert_id or avid from PostgreSQL."""
        # TODO: implement query
        # Example query pattern:
        #   SELECT alert_id, event_code, video_path, ...
        #   FROM alerts WHERE alert_id = %s OR avid = %s
        logger.warning(
            f"PostgresSource.fetch({alert_id}) not yet implemented"
        )
        return None

    def fetch_by_avid(self, avid: str) -> Optional[AlertData]:
        """Fetch alert data by AVID from PostgreSQL."""
        # TODO: implement when AVID support is needed
        logger.warning(
            f"PostgresSource.fetch_by_avid({avid}) not yet implemented"
        )
        return None

    def close(self):
        logger.info("PostgresSource connection closed (stub)")
