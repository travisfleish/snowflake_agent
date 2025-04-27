from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import re
from pydantic import BaseModel, Field, validator, root_validator
from enum import Enum


class QueryExecutionType(str, Enum):
    """Execution type for Snowflake queries."""
    SYNC = "sync"
    ASYNC = "async"


class SnowflakeQueryRequest(BaseModel):
    """Model for Snowflake query request parameters."""

    query: str = Field(
        ...,
        description="SQL query to execute against Snowflake"
    )
    params: Optional[Dict[str, Any]] = Field(
        None,
        description="Optional parameters for query parameterization"
    )
    max_rows: Optional[int] = Field(
        1000,
        description="Maximum number of rows to return"
    )
    timeout: Optional[int] = Field(
        None,
        description="Query timeout in seconds"
    )
    warehouse: Optional[str] = Field(
        None,
        description="Override default warehouse for this query"
    )
    role: Optional[str] = Field(
        None,
        description="Override default role for this query"
    )
    format_output: Optional[bool] = Field(
        True,
        description="Whether to format the output for readability"
    )
    page_size: Optional[int] = Field(
        None,
        description="Results per page if paginating"
    )
    page_token: Optional[str] = Field(
        None,
        description="Token for next page of results"
    )
    execution_type: QueryExecutionType = Field(
        QueryExecutionType.SYNC,
        description="Execution type - synchronous or asynchronous"
    )
    callback_url: Optional[str] = Field(
        None,
        description="URL to call when async query completes"
    )
    notify_on_completion: bool = Field(
        False,
        description="Whether to send notification when async query completes"
    )

    @validator('query')
    def query_must_not_be_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('Query cannot be empty')
        return v.strip()

    @validator('query')
    def validate_sql_safety(cls, v):
        # Check for dangerous operations
        dangerous_operations = ['DROP', 'TRUNCATE', 'DELETE', 'ALTER', 'CREATE',
                                'INSERT', 'UPDATE', 'MERGE', 'GRANT', 'REVOKE']
        for op in dangerous_operations:
            if re.search(fr'\b{op}\b', v.upper()):
                raise ValueError(f"Dangerous operation detected: {op}")

        # Check for SQL injection patterns
        injection_patterns = [
            r';\s*\w+',  # Multiple statements
            r'UNION\s+ALL\s+SELECT',  # UNION-based injection
            r"'\s*OR\s+'1'\s*=\s*'1",  # OR-based injection
            r"'\s*OR\s+1\s*=\s*1",  # OR-based injection
            r'EXEC\s+xp_',  # Stored procedure injection
            r'INTO\s+OUTFILE',  # File writing
        ]

        for pattern in injection_patterns:
            if re.search(pattern, v, re.IGNORECASE):
                raise ValueError("Potential SQL injection detected")

        return v

    @root_validator
    def validate_async_requirements(cls, values):
        execution_type = values.get('execution_type')
        callback_url = values.get('callback_url')

        if execution_type == QueryExecutionType.ASYNC and not callback_url:
            raise ValueError("callback_url is required for asynchronous execution")

        return values


class SnowflakeTableQueryRequest(BaseModel):
    """Model for querying a specific table with filters."""

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
    page_size: Optional[int] = Field(
        None,
        description="Results per page if paginating"
    )
    page_token: Optional[str] = Field(
        None,
        description="Token for next page of results"
    )

    @validator('where_clause')
    def validate_where_clause(cls, v):
        if v:
            # Simple validation to prevent common SQL injection in where clause
            if ";" in v or "--" in v or "/*" in v:
                raise ValueError("Invalid characters in WHERE clause")
        return v

    @validator('table_name', 'schema', 'database')
    def validate_identifier(cls, v):
        if v:
            # Sanitize identifiers to prevent SQL injection
            if not re.match(r'^[A-Za-z0-9_$]+$', v):
                raise ValueError(f"Invalid identifier format: {v}")
        return v


class SnowflakeTableSchemaRequest(BaseModel):
    """Model for table schema inspection request."""

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
    include_statistics: bool = Field(
        False,
        description="Whether to include column statistics"
    )
    include_sample_data: bool = Field(
        False,
        description="Whether to include sample data for each column"
    )

    @validator('table_name', 'schema', 'database')
    def validate_identifier(cls, v):
        if v:
            # Sanitize identifiers to prevent SQL injection
            if not re.match(r'^[A-Za-z0-9_$]+$', v):
                raise ValueError(f"Invalid identifier format: {v}")
        return v


class ColumnStatistics(BaseModel):
    """Model for column statistics."""

    null_count: Optional[int] = None
    distinct_count: Optional[int] = None
    min_value: Optional[Any] = None
    max_value: Optional[Any] = None
    avg_value: Optional[float] = None
    median_value: Optional[Any] = None
    most_common_values: Optional[List[Any]] = None
    histogram: Optional[Dict[str, int]] = None


class SnowflakeColumnInfo(BaseModel):
    """Model for column information in a table schema."""

    name: str = Field(..., description="Column name")
    data_type: str = Field(..., description="Data type of the column")
    nullable: bool = Field(..., description="Whether the column is nullable")
    default: Optional[str] = Field(None, description="Default value for the column")
    primary_key: bool = Field(False, description="Whether the column is a primary key")
    foreign_key: Optional[Dict[str, str]] = Field(
        None,
        description="Foreign key reference information"
    )
    unique: bool = Field(False, description="Whether the column has a unique constraint")
    character_length: Optional[int] = Field(None, description="Length for character types")
    numeric_precision: Optional[int] = Field(None, description="Precision for numeric types")
    numeric_scale: Optional[int] = Field(None, description="Scale for numeric types")
    comment: Optional[str] = Field(None, description="Column comment or description")
    ordinal_position: int = Field(..., description="Position of the column in the table")
    statistics: Optional[ColumnStatistics] = Field(None, description="Statistical information")
    sample_values: Optional[List[Any]] = Field(None, description="Sample values from the column")


class SnowflakeIndexInfo(BaseModel):
    """Model for index information in a table schema."""

    name: str = Field(..., description="Index name")
    columns: List[str] = Field(..., description="Columns in the index")
    unique: bool = Field(False, description="Whether the index is unique")
    index_type: str = Field(..., description="Type of index")
    comment: Optional[str] = Field(None, description="Index comment or description")


class SnowflakeTableSchema(BaseModel):
    """Model for table schema response."""

    table_name: str
    schema_name: str
    database_name: str
    columns: List[SnowflakeColumnInfo]
    primary_key: Optional[List[str]] = None
    foreign_keys: Optional[List[Dict[str, Any]]] = None
    indexes: Optional[List[SnowflakeIndexInfo]] = None
    row_count: Optional[int] = None
    bytes_size: Optional[int] = None
    created_on: Optional[datetime] = None
    last_altered: Optional[datetime] = None
    comment: Optional[str] = None
    table_type: Optional[str] = None
    clustering_key: Optional[List[str]] = None
    retention_time: Optional[int] = None
    search_optimization: Optional[bool] = None
    sample_data: Optional[List[Dict[str, Any]]] = None


class PaginationInfo(BaseModel):
    """Model for pagination details."""

    page_size: int
    current_page: int
    total_rows: int
    total_pages: int
    has_more: bool
    next_page_token: Optional[str] = None


class SnowflakeQueryMetadata(BaseModel):
    """Model for query execution metadata."""

    query_id: str = Field(..., description="Snowflake query ID")
    execution_time: float = Field(..., description="Query execution time in seconds")
    total_rows: int = Field(..., description="Total number of rows returned")
    compilation_time: Optional[float] = Field(None, description="Query compilation time")
    queued_provisioning_time: Optional[float] = Field(
        None,
        description="Time spent waiting for warehouse provisioning"
    )
    queued_repair_time: Optional[float] = Field(
        None,
        description="Time spent in repair queue"
    )
    execution_status: str = Field(..., description="Status of query execution")
    affected_rows: Optional[int] = Field(None, description="Number of rows affected (for DML)")
    warehouse: str = Field(..., description="Warehouse used for execution")
    warehouse_size: Optional[str] = Field(None, description="Size of the warehouse used")
    role: str = Field(..., description="Role used for execution")
    database: str = Field(..., description="Database context for the query")
    schema: str = Field(..., description="Schema context for the query")
    bytes_scanned: Optional[int] = Field(None, description="Bytes scanned during query execution")
    session_id: Optional[str] = Field(None, description="Session ID")
    query_type: Optional[str] = Field(None, description="Type of query executed")
    partitions_scanned: Optional[int] = Field(None, description="Number of partitions scanned")
    partitions_total: Optional[int] = Field(None, description="Total number of partitions")
    execution_time_breakdown: Optional[Dict[str, float]] = Field(
        None,
        description="Detailed breakdown of execution time phases"
    )


class SnowflakeQueryResult(BaseModel):
    """Model for Snowflake query result."""

    data: List[Dict[str, Any]] = Field(
        ...,
        description="Query results as a list of records"
    )
    metadata: SnowflakeQueryMetadata
    column_names: List[str] = Field(..., description="List of column names in result")
    column_types: Dict[str, str] = Field(..., description="Mapping of column names to data types")
    success: bool = Field(..., description="Whether the query executed successfully")
    error_message: Optional[str] = Field(None, description="Error message if query failed")
    truncated: bool = Field(False, description="Whether results were truncated due to limit")
    formatted_data: Optional[str] = Field(None, description="Formatted string representation of data")
    pagination: Optional[PaginationInfo] = Field(None, description="Pagination information")
    execution_type: QueryExecutionType
    async_query_id: Optional[str] = Field(
        None,
        description="ID to retrieve results for async queries"
    )
    query_sql: str = Field(..., description="SQL query that was executed")


class SnowflakeErrorCode(str, Enum):
    """Common Snowflake error codes."""

    SYNTAX_ERROR = "100037"
    INSUFFICIENT_PRIVILEGES = "1043"
    OBJECT_DOES_NOT_EXIST = "2003"
    WAREHOUSE_TIMEOUT = "604"
    NETWORK_ERROR = "101"
    INVALID_PARAMETER = "300001"
    EXECUTION_ERROR = "090001"
    AUTHENTICATION_FAILURE = "250001"
    SESSION_EXPIRED = "390134"
    RESOURCE_LIMIT = "100183"
    SCHEMA_CHANGE = "100080"
    QUERY_TIMEOUT = "604"
    OTHER = "9999"


class SnowflakeErrorResponse(BaseModel):
    """Model for Snowflake error responses."""

    message: str = Field(..., description="Error message")
    code: Optional[SnowflakeErrorCode] = Field(None, description="Error code")
    sql_state: Optional[str] = Field(None, description="SQL state code")
    query_id: Optional[str] = Field(None, description="Associated query ID if available")
    line: Optional[int] = Field(None, description="Line number in query where error occurred")
    position: Optional[int] = Field(None, description="Position in line where error occurred")
    stack_trace: Optional[str] = Field(None, description="Stack trace for server errors")
    timestamp: datetime = Field(default_factory=datetime.now, description="Error timestamp")
    retry_suggested: bool = Field(False, description="Whether a retry might resolve the issue")
    suggestion: Optional[str] = Field(None, description="Suggested action to resolve the error")
    query_sql: Optional[str] = Field(None, description="SQL query that caused the error")
    operation_id: Optional[str] = Field(None, description="ID of the operation that failed")
    tenant_id: Optional[str] = Field(None, description="Snowflake account identifier")


class SnowflakeAsyncQueryStatus(str, Enum):
    """Status values for asynchronous queries."""

    QUEUED = "queued"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    CANCELED = "canceled"
    TIMEOUT = "timeout"


class SnowflakeAsyncQueryStatusResponse(BaseModel):
    """Model for checking the status of an asynchronous query."""

    query_id: str
    status: SnowflakeAsyncQueryStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    estimated_completion_time: Optional[datetime] = None
    percent_complete: Optional[float] = None
    error: Optional[SnowflakeErrorResponse] = None
    warehouse: str
    user: str
    role: str
    database: str
    schema: str
    has_results: bool = False
    can_retrieve_results: bool = False
    query_text: str
    session_id: str