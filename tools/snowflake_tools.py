"""
Snowflake tools for CrewAI.
Provides tool functions that agents can use to interact with Snowflake.
"""

import logging
import json
from typing import Dict, List, Any, Optional, Union, Callable

# Important: Use the basic tool decorator - NOT structured_tool
from langchain.tools import tool, Tool

from utils.snowflake_connector import connector
from utils.snowflake_connector import get_connector
from utils.validators import SQLQueryValidator
from utils.data_processors import SnowflakeDataProcessor

# Configure logger
logger = logging.getLogger(__name__)

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
            if isinstance(input_data, str):
                try:
                    input_dict = json.loads(input_data)
                except json.JSONDecodeError:
                    input_dict = {"query": input_data}
            else:
                input_dict = input_data

            query = input_dict.get("query")
            params = input_dict.get("params")
            max_rows = input_dict.get("max_rows", 1000)
            format_output = input_dict.get("format_output", True)

            is_valid, error_message = SQLQueryValidator.validate_query(query)
            if not is_valid:
                return f"Query validation failed: {error_message}"

            if not SQLQueryValidator.is_read_only(query):
                return "Error: Only read-only (SELECT) queries are allowed for safety"

            if "LIMIT" not in query.upper() and max_rows:
                query = f"{query} LIMIT {max_rows}"

            logger.info(f"Executing Snowflake query: {query[:100]}...")
            results = connector.execute_query(query, params)

            if format_output:
                if not results:
                    return "Query executed successfully. No results returned."

                import pandas as pd
                df = pd.DataFrame(results)
                df = SnowflakeDataProcessor.clean_column_names(df)
                df = SnowflakeDataProcessor.convert_types(df)
                df = SnowflakeDataProcessor.format_dates(df)

                if len(df) > 20:
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
            if isinstance(input_data, str):
                input_dict = json.loads(input_data)
            else:
                input_dict = input_data

            table_name = input_dict.get("table_name")
            columns = input_dict.get("columns")
            where_clause = input_dict.get("where_clause")
            order_by = input_dict.get("order_by")
            limit = input_dict.get("limit", 1000)
            schema = input_dict.get("schema")
            database = input_dict.get("database")

            columns_str = "*"
            if columns:
                columns_str = ", ".join(columns)

            table_name = SQLQueryValidator.sanitize_identifiers(table_name)

            if schema:
                schema = SQLQueryValidator.sanitize_identifiers(schema)
                table_name = f"{schema}.{table_name}"

            if database:
                database = SQLQueryValidator.sanitize_identifiers(database)
                table_name = f"{database}.{table_name}"

            query = f"SELECT {columns_str} FROM {table_name}"

            if where_clause:
                query += f" WHERE {where_clause}"

            if order_by:
                query += f" ORDER BY {order_by}"

            query += f" LIMIT {limit}"

            return SnowflakeQueryTool.execute_query({
                "query": query,
                "max_rows": limit,
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
            if isinstance(input_data, str):
                input_dict = json.loads(input_data)
            else:
                input_dict = input_data

            table_name = input_dict.get("table_name")
            schema = input_dict.get("schema")
            database = input_dict.get("database")

            schema_data = connector.get_table_schema(
                table_name,
                schema,
                database
            )

            if not schema_data:
                return f"No schema information found for table: {table_name}"

            import pandas as pd
            df = pd.DataFrame(schema_data)
            df = SnowflakeDataProcessor.clean_column_names(df)

            return (
                f"Schema for table {table_name}:\n\n"
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
            if isinstance(input_data, str):
                try:
                    input_dict = json.loads(input_data)
                except json.JSONDecodeError:
                    input_dict = {}
            else:
                input_dict = input_data

            schema = input_dict.get("schema")
            database = input_dict.get("database")

            tables = connector.list_tables(schema, database)

            if not tables:
                msg = "No tables found"
                if schema:
                    msg += f" in schema {schema}"
                if database:
                    msg += f" in database {database}"
                return msg

            import pandas as pd
            df = pd.DataFrame(tables)
            df = SnowflakeDataProcessor.clean_column_names(df)

            columns_to_keep = ['table_name', 'table_type', 'row_count', 'created']
            df = df[[col for col in columns_to_keep if col in df.columns]]

            return (
                f"Tables found: {len(tables)}\n\n"
                f"{df.to_string(index=False)}"
            )

        except Exception as e:
            logger.error(f"Error listing tables: {str(e)}")
            return f"Error listing tables: {str(e)}"


# Create simplified tool functions with better input validation
@tool
def execute_snowflake_query(query: str) -> str:
    """
    Execute a SQL query against Snowflake and return the results.
    """
    connector = get_connector()  # ← Dynamically fetch a fresh connector
    if not connector:
        return "Error: No valid Snowflake connector available."

    if not query or not query.strip():
        return "Error: Empty query provided"

    return connector.execute_query(query)

@tool
def query_snowflake_table(table_input: str) -> str:
    """
    Query a Snowflake table with filtering and sorting options.
    """
    if not table_input or not table_input.strip():
        return "Error: No table information provided"

    try:
        json.loads(table_input)
    except json.JSONDecodeError:
        return "Error: Input must be a valid JSON object with table_name at minimum"

    return SnowflakeQueryTool.query_table(table_input)

@tool
def get_snowflake_table_schema(schema_input: str) -> str:
    """
    Retrieve schema information for a Snowflake table.
    """
    if not schema_input or not schema_input.strip():
        return "Error: No table information provided"

    try:
        input_dict = json.loads(schema_input)
        if "table_name" not in input_dict:
            return "Error: table_name is required"
    except json.JSONDecodeError:
        return "Error: Input must be a valid JSON object with table_name"

    return SnowflakeQueryTool.get_table_schema(schema_input)

@tool
def list_snowflake_tables(input_params: str = "{}") -> str:
    """
    List all tables in a Snowflake schema or database.
    """
    if input_params and input_params.strip() != "{}":
        try:
            json.loads(input_params)
        except json.JSONDecodeError:
            return "Error: Input must be a valid JSON object"

    return SnowflakeQueryTool.list_tables(input_params)


# ✅ FINAL fixed get_snowflake_tools()
def get_snowflake_tools():
    """
    Dynamically build Snowflake tools for CrewAI agents.

    Uses StructuredTool's .name and .description attributes.
    """
    tool_functions = [
        execute_snowflake_query,
        query_snowflake_table,
        get_snowflake_table_schema,
        list_snowflake_tables
    ]

    tools = []

    for func in tool_functions:
        tools.append(
            Tool.from_function(
                func=func,
                name=func.name,
                description=func.description
            )
        )

    return tools


# Export classes and helper functions
__all__ = [
    'SnowflakeQueryTool',
    'get_snowflake_tools'
]
