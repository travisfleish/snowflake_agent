"""
Snowflake tools for CrewAI.
Provides tool classes that agents can use to interact with Snowflake.
"""

import logging
import json
from typing import Dict, List, Any, Optional, Union, Callable

from crewai import Tool
from pydantic import BaseModel, Field

from utils.snowflake_connector import connector
from utils.validators import SQLQueryValidator
from utils.data_processors import SnowflakeDataProcessor

# Configure logger
logger = logging.getLogger(__name__)


class QueryInput(BaseModel):
    """Input schema for Snowflake query tool."""

    query: str = Field(
        ...,
        description="SQL query to execute against Snowflake"
    )
    params: Optional[Dict[str, Any]] = Field(
        None,
        description="Optional parameters for the query"
    )
    max_rows: Optional[int] = Field(
        1000,
        description="Maximum number of rows to return"
    )
    format_output: Optional[bool] = Field(
        True,
        description="Whether to format the output for readability"
    )


class TableQueryInput(BaseModel):
    """Input schema for table query tool."""

    table_name: str = Field(
        ...,
        description="Name of the table to query"
    )
    columns: Optional[List[str]] = Field(
        None,
        description="Columns to select (defaults to *)"
    )
    where_clause: Optional[str] = Field(
        None,
        description="WHERE clause for filtering data"
    )
    order_by: Optional[str] = Field(
        None,
        description="ORDER BY clause for sorting data"
    )
    limit: Optional[int] = Field(
        1000,
        description="Maximum number of rows to return"
    )
    schema: Optional[str] = Field(
        None,
        description="Schema name (defaults to connection schema)"
    )
    database: Optional[str] = Field(
        None,
        description="Database name (defaults to connection database)"
    )


class SchemaInput(BaseModel):
    """Input schema for schema inspection tool."""

    table_name: str = Field(
        ...,
        description="Name of the table to inspect"
    )
    schema: Optional[str] = Field(
        None,
        description="Schema name (defaults to connection schema)"
    )
    database: Optional[str] = Field(
        None,
        description="Database name (defaults to connection database)"
    )


class SnowflakeQueryTool:
    """
    Tool for executing Snowflake queries.
    Provides methods for safe query execution and result formatting.
    """

    @staticmethod
    def execute_query(input_data: Union[str, Dict]) -> str:
        """
        Execute a SQL query against Snowflake.

        Args:
            input_data: Query input as JSON string or dictionary

        Returns:
            str: Query results as formatted string
        """
        try:
            # Parse input
            if isinstance(input_data, str):
                try:
                    input_dict = json.loads(input_data)
                except json.JSONDecodeError:
                    # If not valid JSON, assume it's a raw query
                    input_dict = {"query": input_data}
            else:
                input_dict = input_data

            # Create and validate input model
            input_model = QueryInput(**input_dict)

            # Validate query for safety
            is_valid, error_message = SQLQueryValidator.validate_query(input_model.query)
            if not is_valid:
                return f"Query validation failed: {error_message}"

            # Check if query is read-only
            if not SQLQueryValidator.is_read_only(input_model.query):
                return "Error: Only read-only (SELECT) queries are allowed for safety"

            # Add LIMIT clause if not present
            query = input_model.query
            if "LIMIT" not in query.upper() and input_model.max_rows:
                query = f"{query} LIMIT {input_model.max_rows}"

            # Execute query
            logger.info(f"Executing Snowflake query: {query[:100]}...")
            results = connector.execute_query(query, input_model.params)

            # Format results
            if input_model.format_output:
                if not results:
                    return "Query executed successfully. No results returned."

                # Convert to pandas DataFrame for formatting
                import pandas as pd
                df = pd.DataFrame(results)

                # Clean and format
                df = SnowflakeDataProcessor.clean_column_names(df)
                df = SnowflakeDataProcessor.convert_types(df)
                df = SnowflakeDataProcessor.format_dates(df)

                # Convert to string representation
                if len(df) > 20:
                    # Show summary for large result sets
                    return (
                        f"Query returned {len(df)} rows. First 10 rows:\n\n"
                        f"{df.head(10).to_string(index=False)}\n\n"
                        f"...\n\n"
                        f"Last 10 rows:\n\n"
                        f"{df.tail(10).to_string(index=False)}"
                    )
                else:
                    return df.to_string(index=False)
            else:
                # Return raw results
                return json.dumps(results, indent=2, default=str)

        except Exception as e:
            logger.error(f"Error executing Snowflake query: {str(e)}")
            return f"Error executing query: {str(e)}"

    @staticmethod
    def query_table(input_data: Union[str, Dict]) -> str:
        """
        Query data from a specific table with filtering and sorting.

        Args:
            input_data: Table query input as JSON string or dictionary

        Returns:
            str: Query results as formatted string
        """
        try:
            # Parse input
            if isinstance(input_data, str):
                input_dict = json.loads(input_data)
            else:
                input_dict = input_data

            # Create and validate input model
            input_model = TableQueryInput(**input_dict)

            # Build the query
            columns_str = "*"
            if input_model.columns:
                columns_str = ", ".join(input_model.columns)

            # Sanitize table name
            table_name = SQLQueryValidator.sanitize_identifiers(input_model.table_name)

            # Add schema/database if provided
            if input_model.schema:
                schema = SQLQueryValidator.sanitize_identifiers(input_model.schema)
                table_name = f"{schema}.{table_name}"

            if input_model.database:
                database = SQLQueryValidator.sanitize_identifiers(input_model.database)
                table_name = f"{database}.{table_name}"

            # Build query
            query = f"SELECT {columns_str} FROM {table_name}"

            # Add WHERE clause if provided
            if input_model.where_clause:
                query += f" WHERE {input_model.where_clause}"

            # Add ORDER BY clause if provided
            if input_model.order_by:
                query += f" ORDER BY {input_model.order_by}"

            # Add LIMIT clause
            query += f" LIMIT {input_model.limit}"

            # Execute query using the query tool
            return SnowflakeQueryTool.execute_query({
                "query": query,
                "max_rows": input_model.limit,
                "format_output": True
            })

        except Exception as e:
            logger.error(f"Error querying table: {str(e)}")
            return f"Error querying table: {str(e)}"

    @staticmethod
    def get_table_schema(input_data: Union[str, Dict]) -> str:
        """
        Get schema information for a table.

        Args:
            input_data: Schema input as JSON string or dictionary

        Returns:
            str: Table schema as formatted string
        """
        try:
            # Parse input
            if isinstance(input_data, str):
                input_dict = json.loads(input_data)
            else:
                input_dict = input_data

            # Create and validate input model
            input_model = SchemaInput(**input_dict)

            # Get schema information
            schema_data = connector.get_table_schema(
                input_model.table_name,
                input_model.schema,
                input_model.database
            )

            if not schema_data:
                return f"No schema information found for table: {input_model.table_name}"

            # Format as table
            import pandas as pd
            df = pd.DataFrame(schema_data)

            # Clean and format
            df = SnowflakeDataProcessor.clean_column_names(df)

            return (
                f"Schema for table {input_model.table_name}:\n\n"
                f"{df.to_string(index=False)}"
            )

        except Exception as e:
            logger.error(f"Error getting table schema: {str(e)}")
            return f"Error getting table schema: {str(e)}"

    @staticmethod
    def list_tables(input_data: Union[str, Dict]) -> str:
        """
        List all tables in a schema.

        Args:
            input_data: Schema/database input as JSON string or dictionary

        Returns:
            str: List of tables as formatted string
        """
        try:
            # Parse input
            if isinstance(input_data, str):
                try:
                    input_dict = json.loads(input_data)
                except json.JSONDecodeError:
                    # Default to empty dict if not valid JSON
                    input_dict = {}
            else:
                input_dict = input_data

            # Extract schema and database if provided
            schema = input_dict.get("schema")
            database = input_dict.get("database")

            # Get tables
            tables = connector.list_tables(schema, database)

            if not tables:
                msg = "No tables found"
                if schema:
                    msg += f" in schema {schema}"
                if database:
                    msg += f" in database {database}"
                return msg

            # Format as table
            import pandas as pd
            df = pd.DataFrame(tables)

            # Clean and format
            df = SnowflakeDataProcessor.clean_column_names(df)

            # Keep only relevant columns
            columns_to_keep = ['table_name', 'table_type', 'row_count', 'created']
            df = df[[col for col in columns_to_keep if col in df.columns]]

            return (
                f"Tables found: {len(tables)}\n\n"
                f"{df.to_string(index=False)}"
            )

        except Exception as e:
            logger.error(f"Error listing tables: {str(e)}")
            return f"Error listing tables: {str(e)}"


def create_snowflake_tools() -> List[Tool]:
    """
    Create and return Snowflake tools for CrewAI.

    Returns:
        List[Tool]: List of Snowflake tools
    """
    tools = [
        Tool(
            name="execute_snowflake_query",
            description="""
            Execute a SQL query against Snowflake and return the results.

            This tool accepts a SQL query string or a JSON object with the following parameters:
            - query: SQL query to execute (required)
            - params: Dictionary of parameters for the query (optional)
            - max_rows: Maximum number of rows to return (default: 1000)
            - format_output: Whether to format the output for readability (default: true)

            For safety, only SELECT queries are allowed. The tool will automatically add a 
            LIMIT clause if not present in the query.

            Example usage:
            ```
            {
                "query": "SELECT * FROM customers WHERE region = 'WEST'",
                "max_rows": 100,
                "format_output": true
            }
            ```

            Or simply:
            ```
            "SELECT * FROM customers LIMIT 10"
            ```
            """,
            func=SnowflakeQueryTool.execute_query
        ),
        Tool(
            name="query_snowflake_table",
            description="""
            Query data from a specific Snowflake table with filtering and sorting.

            This tool accepts a JSON object with the following parameters:
            - table_name: Name of the table to query (required)
            - columns: List of columns to select (optional, defaults to *)
            - where_clause: WHERE clause for filtering data (optional)
            - order_by: ORDER BY clause for sorting data (optional)
            - limit: Maximum number of rows to return (default: 1000)
            - schema: Schema name (optional, defaults to connection schema)
            - database: Database name (optional, defaults to connection database)

            Example usage:
            ```
            {
                "table_name": "customers",
                "columns": ["customer_id", "name", "email"],
                "where_clause": "region = 'WEST'",
                "order_by": "created_at DESC",
                "limit": 50
            }
            ```
            """,
            func=SnowflakeQueryTool.query_table
        ),
        Tool(
            name="get_snowflake_table_schema",
            description="""
            Get schema information for a Snowflake table.

            This tool accepts a JSON object with the following parameters:
            - table_name: Name of the table to inspect (required)
            - schema: Schema name (optional, defaults to connection schema)
            - database: Database name (optional, defaults to connection database)

            Example usage:
            ```
            {
                "table_name": "customers",
                "schema": "sales"
            }
            ```
            """,
            func=SnowflakeQueryTool.get_table_schema
        ),
        Tool(
            name="list_snowflake_tables",
            description="""
            List all tables in a Snowflake schema.

            This tool accepts a JSON object with the following parameters:
            - schema: Schema name (optional, defaults to connection schema)
            - database: Database name (optional, defaults to connection database)

            Example usage:
            ```
            {
                "schema": "sales",
                "database": "production"
            }
            ```

            Or simply:
            ```
            {}
            ```
            to list tables in the default schema and database.
            """,
            func=SnowflakeQueryTool.list_tables
        )
    ]

    return tools


# Helper functions for direct usage in agents
def get_snowflake_tools() -> List[Tool]:
    """
    Get Snowflake tools for CrewAI agents.

    Returns:
        List[Tool]: List of Snowflake tools
    """
    return create_snowflake_tools()


# Export classes and helper functions
__all__ = [
    'SnowflakeQueryTool',
    'create_snowflake_tools',
    'get_snowflake_tools'
]