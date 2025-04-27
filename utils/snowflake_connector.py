"""
Snowflake connector for the Snowflake Agent application.
Handles database connections, query execution, and result formatting.
"""

import os
import time
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
import contextlib
import base64
import json

import snowflake.connector
from snowflake.connector.connection import SnowflakeConnection
from snowflake.connector.cursor import SnowflakeCursor
from snowflake.connector.errors import (
    ProgrammingError,
    DatabaseError,
    OperationalError,
    InterfaceError
)

from dotenv import load_dotenv
from config.settings import settings

# Configure logger
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class SnowflakeConnector:
    """
    Connector for Snowflake database interactions.
    Provides connection management and query execution functionality.
    """

    def __init__(self,
                 connection_params: Optional[Dict[str, Any]] = None,
                 max_retries: int = 3,
                 retry_delay: int = 2,
                 use_connection_pool: bool = True,
                 pool_size: int = 5):
        """
        Initialize the Snowflake connector.
        """
        # Use provided connection params or load from settings
        self.connection_params = connection_params or settings.snowflake.get_connection_params()
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.use_connection_pool = use_connection_pool
        self.pool_size = pool_size

        # Override user if OAuth is active
        if settings.snowflake.oauth_enabled and hasattr(settings.snowflake, 'override_user'):
            self.connection_params['user'] = settings.snowflake.override_user

        # Connection management
        self._connection = None
        self._connection_pool = []

        # Validate connection parameters
        if not self._validate_connection_params():
            raise ValueError("Invalid Snowflake connection parameters")

        logger.info("SnowflakeConnector initialized")

    def _validate_connection_params(self) -> bool:
        required_params = ['account', 'user']

        for param in required_params:
            if param not in self.connection_params or not self.connection_params[param]:
                logger.error(f"Missing required Snowflake parameter: {param}")
                return False

        if ('password' not in self.connection_params or not self.connection_params['password']) and \
           ('private_key_path' not in self.connection_params or not self.connection_params['private_key_path']) and \
           ('token' not in self.connection_params or not self.connection_params['token']):
            logger.error("Either password, private_key_path, or OAuth token must be provided")
            return False

        return True

    def _create_connection(self) -> SnowflakeConnection:
        try:
            if settings.snowflake.oauth_enabled and settings.snowflake.oauth_token and 'token' not in self.connection_params:
                self.connection_params['token'] = settings.snowflake.oauth_token
                logger.debug("Using OAuth token for Snowflake connection")

            if 'token' in self.connection_params and settings.snowflake.oauth_enabled:
                try:
                    from utils.oauth_manager import oauth_manager
                    user_id = "current_user_id"
                    if not oauth_manager.is_token_valid(user_id):
                        new_token = oauth_manager.get_valid_token(user_id)
                        if new_token:
                            self.connection_params['token'] = new_token
                        else:
                            logger.warning("OAuth token refresh failed, proceeding with current token")
                except (ImportError, AttributeError):
                    logger.warning("OAuth manager not available, proceeding with current token")

            if 'token' in self.connection_params:
                self.connection_params['authenticator'] = 'oauth'
                self.connection_params.pop('password', None)

            connection = snowflake.connector.connect(**self.connection_params)
            logger.debug("Created new Snowflake connection")
            return connection
        except (DatabaseError, OperationalError, InterfaceError) as e:
            logger.error(f"Failed to connect to Snowflake: {str(e)}")
            raise ConnectionError(f"Failed to connect to Snowflake: {str(e)}")

    def _get_connection_from_pool(self) -> Optional[SnowflakeConnection]:
        if not self._connection_pool:
            return None

        connection = self._connection_pool.pop()

        try:
            if connection.is_closed():
                logger.debug("Discarding closed connection from pool")
                return self._get_connection_from_pool()
        except Exception:
            logger.debug("Discarding invalid connection from pool")
            return self._get_connection_from_pool()

        return connection

    def _return_connection_to_pool(self, connection: SnowflakeConnection) -> None:
        if len(self._connection_pool) < self.pool_size:
            self._connection_pool.append(connection)
        else:
            connection.close()

    def get_connection(self) -> SnowflakeConnection:
        if self.use_connection_pool:
            connection = self._get_connection_from_pool()
            if connection is not None:
                return connection

        return self._create_connection()

    @contextlib.contextmanager
    def connection(self) -> SnowflakeConnection:
        connection = self.get_connection()
        try:
            yield connection
        finally:
            if self.use_connection_pool:
                self._return_connection_to_pool(connection)
            else:
                connection.close()

    def execute_query(self, query: str, params: Optional[Dict[str, Any]] = None,
                      return_cursor: bool = False) -> Union[List[Dict[str, Any]], SnowflakeCursor]:
        retries = 0
        last_error = None

        while retries <= self.max_retries:
            try:
                with self.connection() as conn:
                    cursor = conn.cursor()

                    if params:
                        cursor.execute(query, params)
                    else:
                        cursor.execute(query)

                    if return_cursor:
                        return cursor

                    if cursor.description:
                        columns = [col[0] for col in cursor.description]
                        results = []

                        for row in cursor:
                            results.append(dict(zip(columns, row)))

                        cursor.close()
                        return results
                    else:
                        cursor.close()
                        return []

            except (DatabaseError, OperationalError, ProgrammingError) as e:
                last_error = e
                retries += 1

                if settings.snowflake.oauth_enabled and isinstance(e, DatabaseError) and "JWT token has expired" in str(e):
                    logger.warning("OAuth token expired during query execution, attempting refresh")
                    try:
                        from utils.oauth_manager import oauth_manager
                        user_id = "current_user_id"
                        if oauth_manager.refresh_token(user_id):
                            self.connection_params['token'] = settings.snowflake.oauth_token
                    except (ImportError, AttributeError):
                        logger.warning("OAuth manager not available for token refresh")

                if retries <= self.max_retries:
                    logger.warning(f"Query execution failed, retrying ({retries}/{self.max_retries}): {str(e)}")
                    time.sleep(self.retry_delay)
                else:
                    logger.error(f"Query execution failed after {self.max_retries} retries: {str(e)}")

        raise last_error

    def execute_queries(self, queries: List[str]) -> List[List[Dict[str, Any]]]:
        results = []

        for query in queries:
            result = self.execute_query(query)
            results.append(result)

        return results

    def get_table_schema(self, table_name: str,
                         schema: Optional[str] = None,
                         database: Optional[str] = None) -> List[Dict[str, Any]]:
        schema = schema or self.connection_params.get('schema')
        database = database or self.connection_params.get('database')

        query = """
        SELECT 
            COLUMN_NAME, 
            DATA_TYPE,
            CHARACTER_MAXIMUM_LENGTH,
            NUMERIC_PRECISION,
            NUMERIC_SCALE,
            IS_NULLABLE,
            COLUMN_DEFAULT,
            COMMENT
        FROM 
            INFORMATION_SCHEMA.COLUMNS
        WHERE 
            TABLE_NAME = %s
        """

        params = {'table_name': table_name}

        if schema:
            query += " AND TABLE_SCHEMA = %s"
            params['schema'] = schema

        if database:
            query += " AND TABLE_CATALOG = %s"
            params['database'] = database

        query += " ORDER BY ORDINAL_POSITION"

        return self.execute_query(query, params)

    def list_tables(self, schema: Optional[str] = None,
                    database: Optional[str] = None) -> List[Dict[str, Any]]:
        schema = schema or self.connection_params.get('schema')
        database = database or self.connection_params.get('database')

        query = """
        SELECT 
            TABLE_NAME,
            TABLE_TYPE,
            ROW_COUNT,
            BYTES,
            CREATED,
            LAST_ALTERED,
            COMMENT
        FROM 
            INFORMATION_SCHEMA.TABLES
        WHERE 
            1=1
        """

        params = {}

        if schema:
            query += " AND TABLE_SCHEMA = %s"
            params['schema'] = schema

        if database:
            query += " AND TABLE_CATALOG = %s"
            params['database'] = database

        query += " ORDER BY TABLE_NAME"

        return self.execute_query(query, params)

    def test_connection(self) -> bool:
        try:
            with self.connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT 1")
                cursor.close()
            return True
        except Exception as e:
            logger.error(f"Connection test failed: {str(e)}")
            return False

    def close_all_connections(self) -> None:
        if self._connection is not None:
            try:
                self._connection.close()
            except Exception as e:
                logger.error(f"Error closing connection: {str(e)}")
            self._connection = None

        for conn in self._connection_pool:
            try:
                conn.close()
            except Exception as e:
                logger.error(f"Error closing pooled connection: {str(e)}")

        self._connection_pool = []
        logger.info("All connections closed")


# Global connector instance
_connector_instance = None

def decode_sub_from_token(token: str) -> str:
    """Extract sub (subject) from OAuth token."""
    try:
        parts = token.split('.')
        payload_base64 = parts[1]
        padding = '=' * (4 - len(payload_base64) % 4)
        payload_json = base64.urlsafe_b64decode(payload_base64 + padding)
        payload = json.loads(payload_json)
        return payload.get('sub')
    except Exception as e:
        raise ValueError(f"Failed to decode token: {e}")

def get_connector():
    global _connector_instance
    if _connector_instance is None:
        try:
            _connector_instance = SnowflakeConnector()
        except ValueError as e:
            logger.warning(f"Could not initialize connector: {str(e)}")
            print(f"❌ Connector init failed: {str(e)}")
            return None
    print(f"✅ Connector initialized successfully")
    return _connector_instance

def init_with_oauth_token(token):
    """
    Initialize Snowflake connector with an OAuth token.
    """
    global _connector_instance

    sub = decode_sub_from_token(token)

    settings.snowflake.oauth_token = token
    settings.snowflake.oauth_enabled = True
    settings.snowflake.override_user = sub

    try:
        _connector_instance = SnowflakeConnector()
        return _connector_instance
    except ValueError as e:
        logger.error(f"Error initializing connector with OAuth token: {str(e)}")
        return None

# Helper functions for direct usage
def execute_snowflake_query(query: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    connector = get_connector()
    if connector is None:
        return [{"error": "Snowflake connector is not available"}]
    return connector.execute_query(query, params)

def get_table_data(table_name: str,
                   limit: int = 1000,
                   where_clause: Optional[str] = None,
                   order_by: Optional[str] = None) -> List[Dict[str, Any]]:
    connector = get_connector()
    if connector is None:
        return [{"error": "Snowflake connector is not available"}]

    query = f"SELECT * FROM {table_name}"

    if where_clause:
        query += f" WHERE {where_clause}"

    if order_by:
        query += f" ORDER BY {order_by}"

    query += f" LIMIT {limit}"

    return connector.execute_query(query)

# Export functions
connector = get_connector()

__all__ = [
    'SnowflakeConnector',
    'get_connector',
    'init_with_oauth_token',
    'execute_snowflake_query',
    'get_table_data',
    'connector'
]
