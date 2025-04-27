import pytest
from unittest.mock import Mock, patch, MagicMock
import json
import pandas as pd
from typing import Dict, List, Any, Optional

from tools.snowflake_tools import SnowflakeQueryTool, QueryInput
from utils.snowflake_connector import connector
from utils.validators import SQLQueryValidator
from utils.data_processors import SnowflakeDataProcessor


# Fixtures for common test setup
@pytest.fixture
def mock_connector():
    """Mock the Snowflake connector."""
    with patch('tools.snowflake_tools.connector') as mock_conn:
        # Configure the mock
        mock_cursor = MagicMock()
        mock_cursor.description = [('col1',), ('col2',)]
        mock_cursor.fetchall.return_value = [(1, 'a'), (2, 'b')]

        mock_conn.execute_query.return_value = [
            {"col1": 1, "col2": "a"},
            {"col1": 2, "col2": "b"}
        ]

        yield mock_conn


@pytest.fixture
def mock_validator():
    """Mock the SQL validator."""
    with patch('tools.snowflake_tools.SQLQueryValidator') as mock_val:
        # Configure the mock
        mock_val.validate_query.return_value = (True, "Query validated successfully")
        mock_val.is_read_only.return_value = True
        mock_val.sanitize_identifiers.side_effect = lambda x: x  # Return input unchanged

        yield mock_val


@pytest.fixture
def mock_data_processor():
    """Mock the data processor."""
    with patch('tools.snowflake_tools.SnowflakeDataProcessor') as mock_proc:
        # Configure the mock
        mock_proc.clean_column_names.side_effect = lambda df: df
        mock_proc.convert_types.side_effect = lambda df: df
        mock_proc.format_dates.side_effect = lambda df: df

        yield mock_proc


@pytest.fixture
def query_tool():
    """Create a SnowflakeQueryTool instance."""
    return SnowflakeQueryTool()


# Test cases
class TestSnowflakeQueryTool:
    def test_execute_query_with_string_input(self, query_tool, mock_connector, mock_validator):
        """Test executing a query with a string input."""
        # Test data
        test_query = "SELECT * FROM test_table"

        # Call the function
        result = query_tool.execute_query(test_query)

        # Assertions
        mock_validator.validate_query.assert_called_once()
        mock_validator.is_read_only.assert_called_once_with(test_query)
        mock_connector.execute_query.assert_called_once()
        assert len(result) == 2
        assert result[0]["col1"] == 1
        assert result[0]["col2"] == "a"

    def test_execute_query_with_dict_input(self, query_tool, mock_connector, mock_validator):
        """Test executing a query with a dictionary input."""
        # Test data
        test_input = {
            "query": "SELECT * FROM test_table",
            "params": {"param1": "value1"},
            "max_rows": 500,
            "format_output": True
        }

        # Call the function
        result = query_tool.execute_query(test_input)

        # Assertions
        mock_validator.validate_query.assert_called_once()
        mock_validator.is_read_only.assert_called_once()
        mock_connector.execute_query.assert_called_once()
        assert len(result) == 2

    def test_execute_query_adds_limit_clause(self, query_tool, mock_connector, mock_validator):
        """Test that LIMIT clause is added to the query if not present."""
        # Test data
        test_query = "SELECT * FROM test_table"

        # Call the function
        query_tool.execute_query(test_query)

        # Get the actual query passed to execute_query
        actual_query = mock_connector.execute_query.call_args[0][0]

        # Assertions
        assert "LIMIT 1000" in actual_query

    def test_execute_query_preserves_existing_limit(self, query_tool, mock_connector, mock_validator):
        """Test that existing LIMIT clause is preserved."""
        # Test data
        test_query = "SELECT * FROM test_table LIMIT 50"

        # Call the function
        query_tool.execute_query(test_query)

        # Get the actual query passed to execute_query
        actual_query = mock_connector.execute_query.call_args[0][0]

        # Assertions
        assert actual_query == test_query  # Should be unchanged

    def test_execute_query_with_params(self, query_tool, mock_connector, mock_validator):
        """Test executing a query with parameters."""
        # Test data
        test_input = {
            "query": "SELECT * FROM test_table WHERE col1 = %s",
            "params": {"col1": 1}
        }

        # Call the function
        query_tool.execute_query(test_input)

        # Assertions
        mock_connector.execute_query.assert_called_once_with(
            "SELECT * FROM test_table WHERE col1 = %s LIMIT 1000",
            {"col1": 1}
        )

    def test_execute_query_validation_failure(self, query_tool, mock_connector, mock_validator):
        """Test query validation failure."""
        # Configure mock to fail validation
        mock_validator.validate_query.return_value = (False, "Validation failed")

        # Test data
        test_query = "SELECT * FROM test_table"

        # Call the function
        result = query_tool.execute_query(test_query)

        # Assertions
        assert "Validation failed" in result
        mock_connector.execute_query.assert_not_called()

    def test_execute_query_non_readonly_query(self, query_tool, mock_connector, mock_validator):
        """Test rejection of non-read-only queries."""
        # Configure mock to indicate non-read-only query
        mock_validator.is_read_only.return_value = False

        # Test data
        test_query = "DELETE FROM test_table"

        # Call the function
        result = query_tool.execute_query(test_query)

        # Assertions
        assert "Only read-only (SELECT) queries are allowed" in result
        mock_connector.execute_query.assert_not_called()

    def test_execute_query_formats_output(self, query_tool, mock_connector, mock_validator, mock_data_processor):
        """Test output formatting for query results."""
        # Test data
        test_input = {
            "query": "SELECT * FROM test_table",
            "format_output": True
        }

        # Configure mock to return DataFrame for to_string
        with patch('pandas.DataFrame') as mock_df_class:
            mock_df = MagicMock()
            mock_df.head.return_value = mock_df
            mock_df.tail.return_value = mock_df
            mock_df.to_string.return_value = "formatted output"
            mock_df_class.return_value = mock_df

            # Call the function
            result = query_tool.execute_query(test_input)

            # Assertions
            mock_df_class.assert_called_once()
            assert result == "formatted output"

    def test_execute_query_raw_output(self, query_tool, mock_connector, mock_validator):
        """Test raw JSON output for query results."""
        # Test data
        test_input = {
            "query": "SELECT * FROM test_table",
            "format_output": False
        }

        # Call the function
        result = query_tool.execute_query(test_input)

        # Parse JSON result
        parsed = json.loads(result)

        # Assertions
        assert isinstance(parsed, list)
        assert len(parsed) == 2
        assert parsed[0]["col1"] == 1
        assert parsed[0]["col2"] == "a"

    def test_execute_query_empty_results(self, query_tool, mock_connector, mock_validator):
        """Test handling of empty query results."""
        # Configure mock to return empty results
        mock_connector.execute_query.return_value = []

        # Test data
        test_query = "SELECT * FROM empty_table"

        # Call the function
        result = query_tool.execute_query(test_query)

        # Assertions
        assert "No results" in result

    def test_execute_query_with_error(self, query_tool, mock_connector, mock_validator):
        """Test error handling during query execution."""
        # Configure mock to raise exception
        mock_connector.execute_query.side_effect = Exception("Test error")

        # Test data
        test_query = "SELECT * FROM test_table"

        # Call the function
        result = query_tool.execute_query(test_query)

        # Assertions
        assert "Error executing query" in result
        assert "Test error" in result

    def test_query_table_basic(self, query_tool, mock_connector, mock_validator):
        """Test the query_table method with basic parameters."""
        # Mock execute_query to track calls
        with patch.object(SnowflakeQueryTool, 'execute_query', return_value="mocked result") as mock_execute:
            # Test data
            test_input = {
                "table_name": "test_table",
                "columns": ["col1", "col2"],
                "where_clause": "col1 > 0",
                "order_by": "col1 DESC",
                "limit": 50
            }

            # Call the function
            result = query_tool.query_table(test_input)

            # Assertions
            mock_execute.assert_called_once()
            expected_query = "SELECT col1, col2 FROM test_table WHERE col1 > 0 ORDER BY col1 DESC LIMIT 50"
            actual_query = mock_execute.call_args[0][0]["query"]
            assert expected_query == actual_query
            assert result == "mocked result"

    def test_query_table_with_schema_db(self, query_tool, mock_connector, mock_validator):
        """Test query_table with schema and database specified."""
        # Mock execute_query to track calls
        with patch.object(SnowflakeQueryTool, 'execute_query', return_value="mocked result") as mock_execute:
            # Test data
            test_input = {
                "table_name": "test_table",
                "schema": "test_schema",
                "database": "test_db"
            }

            # Call the function
            result = query_tool.query_table(test_input)

            # Assertions
            mock_execute.assert_called_once()
            expected_query = "SELECT * FROM test_db.test_schema.test_table LIMIT 1000"
            actual_query = mock_execute.call_args[0][0]["query"]
            assert expected_query == actual_query

    def test_query_table_json_input(self, query_tool, mock_connector, mock_validator):
        """Test query_table with JSON string input."""
        # Mock execute_query to track calls
        with patch.object(SnowflakeQueryTool, 'execute_query', return_value="mocked result") as mock_execute:
            # Test data
            test_input = json.dumps({
                "table_name": "test_table",
                "limit": 50
            })

            # Call the function
            result = query_tool.query_table(test_input)

            # Assertions
            mock_execute.assert_called_once()
            expected_query = "SELECT * FROM test_table LIMIT 50"
            actual_query = mock_execute.call_args[0][0]["query"]
            assert expected_query == actual_query

    def test_query_table_error_handling(self, query_tool, mock_connector, mock_validator):
        """Test error handling in query_table."""
        # Mock execute_query to raise exception
        with patch.object(SnowflakeQueryTool, 'execute_query', side_effect=Exception("Test error")) as mock_execute:
            # Test data
            test_input = {"table_name": "test_table"}

            # Call the function
            result = query_tool.query_table(test_input)

            # Assertions
            assert "Error querying table" in result
            assert "Test error" in result

    def test_get_table_schema(self, query_tool, mock_connector, mock_validator):
        """Test the get_table_schema method."""
        # Configure mock_connector.get_table_schema
        mock_connector.get_table_schema.return_value = [
            {"COLUMN_NAME": "col1", "DATA_TYPE": "INT", "IS_NULLABLE": "NO"},
            {"COLUMN_NAME": "col2", "DATA_TYPE": "VARCHAR", "IS_NULLABLE": "YES"}
        ]

        # Mock pandas DataFrame for formatting
        with patch('pandas.DataFrame') as mock_df_class:
            mock_df = MagicMock()
            mock_df.to_string.return_value = "formatted schema"
            mock_df_class.return_value = mock_df

            # Test data
            test_input = {
                "table_name": "test_table",
                "schema": "test_schema",
                "database": "test_db"
            }

            # Call the function
            result = query_tool.get_table_schema(test_input)

            # Assertions
            mock_connector.get_table_schema.assert_called_once_with(
                "test_table", "test_schema", "test_db"
            )
            assert "Schema for table test_table" in result
            assert "formatted schema" in result

    def test_get_table_schema_no_results(self, query_tool, mock_connector, mock_validator):
        """Test get_table_schema with no schema information found."""
        # Configure mock to return empty results
        mock_connector.get_table_schema.return_value = []

        # Test data
        test_input = {"table_name": "nonexistent_table"}

        # Call the function
        result = query_tool.get_table_schema(test_input)

        # Assertions
        assert "No schema information found" in result

    def test_get_table_schema_error(self, query_tool, mock_connector, mock_validator):
        """Test error handling in get_table_schema."""
        # Configure mock to raise exception
        mock_connector.get_table_schema.side_effect = Exception("Test error")

        # Test data
        test_input = {"table_name": "test_table"}

        # Call the function
        result = query_tool.get_table_schema(test_input)

        # Assertions
        assert "Error getting table schema" in result
        assert "Test error" in result

    def test_list_tables(self, query_tool, mock_connector, mock_validator):
        """Test the list_tables method."""
        # Configure mock_connector.list_tables
        mock_connector.list_tables.return_value = [
            {"table_name": "table1", "table_type": "TABLE", "row_count": 100},
            {"table_name": "table2", "table_type": "VIEW", "row_count": 50}
        ]

        # Mock pandas DataFrame for formatting
        with patch('pandas.DataFrame') as mock_df_class:
            mock_df = MagicMock()
            mock_df.to_string.return_value = "formatted tables list"
            mock_df_class.return_value = mock_df

            # Test data
            test_input = {
                "schema": "test_schema",
                "database": "test_db"
            }

            # Call the function
            result = query_tool.list_tables(test_input)

            # Assertions
            mock_connector.list_tables.assert_called_once_with("test_schema", "test_db")
            assert "Tables found: 2" in result
            assert "formatted tables list" in result

    def test_list_tables_empty(self, query_tool, mock_connector, mock_validator):
        """Test list_tables with no tables found."""
        # Configure mock to return empty results
        mock_connector.list_tables.return_value = []

        # Test data
        test_input = {"schema": "empty_schema"}

        # Call the function
        result = query_tool.list_tables(test_input)

        # Assertions
        assert "No tables found in schema empty_schema" in result

    def test_list_tables_error(self, query_tool, mock_connector, mock_validator):
        """Test error handling in list_tables."""
        # Configure mock to raise exception
        mock_connector.list_tables.side_effect = Exception("Test error")

        # Test data
        test_input = {"schema": "test_schema"}

        # Call the function
        result = query_tool.list_tables(test_input)

        # Assertions
        assert "Error listing tables" in result
        assert "Test error" in result

    def test_create_snowflake_tools(self):
        """Test the create_snowflake_tools function."""
        from tools.snowflake_tools import create_snowflake_tools

        # Call the function
        tools = create_snowflake_tools()

        # Assertions
        assert len(tools) == 4
        tool_names = [tool.name for tool in tools]
        assert "execute_snowflake_query" in tool_names
        assert "query_snowflake_table" in tool_names
        assert "get_snowflake_table_schema" in tool_names
        assert "list_snowflake_tables" in tool_names

    def test_get_snowflake_tools(self):
        """Test the get_snowflake_tools function."""
        from tools.snowflake_tools import get_snowflake_tools, create_snowflake_tools

        # Mock create_snowflake_tools
        with patch('tools.snowflake_tools.create_snowflake_tools', return_value=["tool1", "tool2"]) as mock_create:
            # Call the function
            tools = get_snowflake_tools()

            # Assertions
            mock_create.assert_called_once()
            assert tools == ["tool1", "tool2"]