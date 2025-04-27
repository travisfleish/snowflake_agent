"""
Snowflake connector for the Snowflake Agent application.
Handles database connections, query execution, and result formatting.
"""

import os
import time
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
import contextlib

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

        Args:
            connection_params: Optional custom connection parameters
            max_retries: Maximum number of retry attempts for failed queries
            retry_delay: Delay between retry attempts in seconds
            use_connection_pool: Whether to use connection pooling
            pool_size: Size of the connection pool
        """
        # Use provided connection params or load from settings
        self.connection_params = connection_params or settings.snowflake.get_connection_params()
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.use_connection_pool = use_connection_pool
        self.pool_size = pool_size

        # Connection management
        self._connection = None
        self._connection_pool = []

        # Validate connection parameters
        if not self._validate_connection_params():
            raise ValueError("Invalid Snowflake connection parameters")

        logger.info("SnowflakeConnector initialized")

    def _validate_connection_params(self) -> bool:
        """
        Validate that the required connection parameters are present.

        Returns:
            bool: True if parameters are valid, False otherwise
        """
        required_params = ['account', 'user']

        # Check required parameters
        for param in required_params:
            if param not in self.connection_params or not self.connection_params[param]:
                logger.error(f"Missing required Snowflake parameter: {param}")
                return False

        # Check authentication method
        if ('password' not in self.connection_params or not self.connection_params['password']) and \
           ('private_key_path' not in self.connection_params or not self.connection_params['private_key_path']) and \
           ('token' not in self.connection_params or not self.connection_params['token']):
            logger.error("Either password, private_key_path, or OAuth token must be provided")
            return False

        return True

    def _create_connection(self) -> SnowflakeConnection:
        """
        Create a new Snowflake connection with OAuth support.

        Returns:
            SnowflakeConnection: Snowflake connection object

        Raises:
            ConnectionError: If connection fails
        """
        try:
            # If OAuth is enabled, ensure we have a valid token
            if settings.snowflake.oauth_enabled and settings.snowflake.oauth_token and 'token' not in self.connection_params:
                # Update connection params with the token
                self.connection_params['token'] = settings.snowflake.oauth_token
                logger.debug("Using OAuth token for Snowflake connection")

            # Check if we need to refresh an OAuth token
            if 'token' in self.connection_params and settings.snowflake.oauth_enabled:
                # In a real application, you'd get the user ID from the session
                # This is just a placeholder - you would implement user session management
                try:
                    from utils.oauth_manager import oauth_manager
                    user_id = "current_user_id"  # This should come from your session management
                    if not oauth_manager.is_token_valid(user_id):
                        new_token = oauth_manager.get_valid_token(user_id)
                        if new_token:
                            self.connection_params['token'] = new_token
                        else:
                            logger.warning("OAuth token refresh failed, proceeding with current token")
                except (ImportError, AttributeError):
                    logger.warning("OAuth manager not available, proceeding with current token")

            connection = snowflake.connector.connect(**self.connection_params)
            logger.debug("Created new Snowflake connection")
            return connection
        except (DatabaseError, OperationalError, InterfaceError) as e:
            logger.error(f"Failed to connect to Snowflake: {str(e)}")
            raise ConnectionError(f"Failed to connect to Snowflake: {str(e)}")

    def _get_connection_from_pool(self) -> Optional[SnowflakeConnection]:
        """
        Get an available connection from the pool.

        Returns:
            Optional[SnowflakeConnection]: Connection from pool or None if pool is empty
        """
        if not self._connection_pool:
            return None

        connection = self._connection_pool.pop()

        # Check if connection is still valid
        try:
            if connection.is_closed():
                logger.debug("Discarding closed connection from pool")
                return self._get_connection_from_pool()
        except Exception:
            logger.debug("Discarding invalid connection from pool")
            return self._get_connection_from_pool()

        return connection

    def _return_connection_to_pool(self, connection: SnowflakeConnection) -> None:
        """
        Return a connection to the pool.

        Args:
            connection: Snowflake connection object
        """
        if len(self._connection_pool) < self.pool_size:
            self._connection_pool.append(connection)
        else:
            connection.close()

    def get_connection(self) -> SnowflakeConnection:
        """
        Get a Snowflake connection (from pool or create new).

        Returns:
            SnowflakeConnection: Snowflake connection object
        """
        if self.use_connection_pool:
            # Try to get connection from pool
            connection = self._get_connection_from_pool()
            if connection is not None:
                return connection

        # Create new connection
        return self._create_connection()

    @contextlib.contextmanager
    def connection(self) -> SnowflakeConnection:
        """
        Context manager for Snowflake connections.

        Yields:
            SnowflakeConnection: Snowflake connection object

        Example:
            with connector.connection() as conn:
                # Use connection
                pass
        """
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
        """
        Execute a SQL query and return results.

        Args:
            query: SQL query to execute
            params: Optional query parameters
            return_cursor: Whether to return the cursor instead of results

        Returns:
            Union[List[Dict[str, Any]], SnowflakeCursor]: Query results or cursor

        Raises:
            Exception: If query execution fails
        """
        retries = 0
        last_error = None

        while retries <= self.max_retries:
            try:
                with self.connection() as conn:
                    cursor = conn.cursor()

                    # Execute query with parameters if provided
                    if params:
                        cursor.execute(query, params)
                    else:
                        cursor.execute(query)

                    # Return cursor if requested
                    if return_cursor:
                        return cursor

                    # Convert results to list of dictionaries
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

                # Check if the error is due to an expired OAuth token
                if settings.snowflake.oauth_enabled and isinstance(e, DatabaseError) and "JWT token has expired" in str(e):
                    logger.warning("OAuth token expired during query execution, attempting refresh")
                    try:
                        from utils.oauth_manager import oauth_manager
                        user_id = "current_user_id"  # This should come from your session management
                        if oauth_manager.refresh_token(user_id):
                            # Update connection params with new token
                            self.connection_params['token'] = settings.snowflake.oauth_token
                    except (ImportError, AttributeError):
                        logger.warning("OAuth manager not available for token refresh")

                # Log retry attempt
                if retries <= self.max_retries:
                    logger.warning(f"Query execution failed, retrying ({retries}/{self.max_retries}): {str(e)}")
                    time.sleep(self.retry_delay)
                else:
                    logger.error(f"Query execution failed after {self.max_retries} retries: {str(e)}")

        # If we've exhausted retries, raise the last error
        raise last_error

    def execute_queries(self, queries: List[str]) -> List[List[Dict[str, Any]]]:
        """
        Execute multiple SQL queries in sequence.

        Args:
            queries: List of SQL queries to execute

        Returns:
            List[List[Dict[str, Any]]]: List of query results

        Raises:
            Exception: If any query execution fails
        """
        results = []

        for query in queries:
            result = self.execute_query(query)
            results.append(result)

        return results

    def get_table_schema(self, table_name: str,
                         schema: Optional[str] = None,
                         database: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get the schema information for a table.

        Args:
            table_name: Name of the table
            schema: Optional schema name (defaults to connection schema)
            database: Optional database name (defaults to connection database)

        Returns:
            List[Dict[str, Any]]: Table schema information
        """
        # Use provided values or connection defaults
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
        """
        List all tables in a schema.

        Args:
            schema: Optional schema name (defaults to connection schema)
            database: Optional database name (defaults to connection database)

        Returns:
            List[Dict[str, Any]]: List of tables
        """
        # Use provided values or connection defaults
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
        """
        Test the Snowflake connection.

        Returns:
            bool: True if connection is successful, False otherwise
        """
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
        """
        Close all pooled connections.
        """
        if self._connection is not None:
            try:
                self._connection.close()
            except Exception as e:
                logger.error(f"Error closing connection: {str(e)}")
            self._connection = None

        # Close all pooled connections
        for conn in self._connection_pool:
            try:
                conn.close()
            except Exception as e:
                logger.error(f"Error closing pooled connection: {str(e)}")

        self._connection_pool = []
        logger.info("All connections closed")


# Create a default connector instance
connector = SnowflakeConnector()


# Helper functions for direct usage
def execute_snowflake_query(query: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """
    Execute a Snowflake query and return results as list of dictionaries.

    Args:
        query: SQL query to execute
        params: Optional query parameters

    Returns:
        List[Dict[str, Any]]: Query results
    """
    return connector.execute_query(query, params)


def get_table_data(table_name: str,
                   limit: int = 1000,
                   where_clause: Optional[str] = None,
                   order_by: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Get data from a table with optional filtering and ordering.

    Args:
        table_name: Name of the table
        limit: Maximum number of rows to return
        where_clause: Optional WHERE clause
        order_by: Optional ORDER BY clause

    Returns:
        List[Dict[str, Any]]: Table data
    """
    query = f"SELECT * FROM {table_name}"

    if where_clause:
        query += f" WHERE {where_clause}"

    if order_by:
        query += f" ORDER BY {order_by}"

    query += f" LIMIT {limit}"

    return connector.execute_query(query)


# Export classes and helper functions
__all__ = [
    'SnowflakeConnector',
    'connector',
    'execute_snowflake_query',
    'get_table_data'
]