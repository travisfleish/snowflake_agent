"""
Connection manager for Snowflake with lazy initialization.
"""

import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class LazySnowflakeConnector:
    """
    Lazy connector that only initializes a connection when needed.
    """

    def __init__(self):
        """Initialize the lazy connector."""
        self._connector = None
        self._init_attempted = False

    def _initialize(self) -> bool:
        """Attempt to initialize the connector."""
        if self._init_attempted:
            return self._connector is not None

        self._init_attempted = True

        try:
            # Import here to avoid circular imports
            from utils.snowflake_connector import get_connector

            # Get the connector
            self._connector = get_connector()
            return self._connector is not None

        except Exception as e:
            logger.error(f"Error initializing Snowflake connector: {str(e)}")
            return False

    def is_available(self) -> bool:
        """Check if Snowflake connection is available."""
        return self._initialize() and self._connector is not None

    def test_connection(self) -> bool:
        """Test the Snowflake connection."""
        if not self._initialize():
            return False

        try:
            return self._connector.test_connection()
        except Exception as e:
            logger.error(f"Error testing Snowflake connection: {str(e)}")
            return False

    def get_connector(self):
        """Get the actual connector instance, initializing if needed."""
        if not self._initialize():
            raise ValueError("Snowflake connector is not available. Check your configuration.")

        return self._connector

    def execute_query(self, *args, **kwargs):
        """Execute a query, initializing the connector if needed."""
        if not self._initialize():
            return {"error": "Snowflake connection not available. Check configuration."}

        return self._connector.execute_query(*args, **kwargs)

    def setup_oauth(self):
        """Set up OAuth authentication for Snowflake."""
        from app.oauth_setup import setup_snowflake_with_oauth

        # Reset initialization flag
        self._init_attempted = False
        self._connector = None

        # Attempt OAuth setup
        return setup_snowflake_with_oauth()


# Create singleton instance
lazy_connector = LazySnowflakeConnector()