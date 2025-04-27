"""
SQL and request validators for Snowflake Agent.
Provides validation functions to ensure safe query execution and request handling.
"""

import re
import json
from typing import Dict, List, Any, Union, Optional, Tuple, Set


class SQLQueryValidator:
    """
    SQL query validator to ensure safe execution of user inputs.
    Prevents SQL injection and other potentially harmful operations.
    """

    # Dangerous SQL operations that should be restricted
    DANGEROUS_OPERATIONS = {
        'DROP', 'TRUNCATE', 'DELETE', 'ALTER', 'CREATE', 'INSERT', 'UPDATE',
        'MERGE', 'GRANT', 'REVOKE', 'REPLACE', 'EXECUTE', 'CALL', 'COPY'
    }

    # SQL comments that could be used to hide malicious code
    COMMENT_PATTERNS = [
        r'--.*?(\n|$)',  # Single line comments
        r'/\*.*?\*/',  # Multi-line comments
    ]

    # Patterns for detecting potential SQL injection attempts
    INJECTION_PATTERNS = [
        r';\s*\w+',  # Multiple statements (e.g., SELECT 1; DROP TABLE)
        r'UNION\s+ALL\s+SELECT',  # UNION-based injection
        r'UNION\s+SELECT',  # UNION-based injection
        r"'\s*OR\s+'1'\s*=\s*'1",  # OR-based injection
        r"'\s*OR\s+1\s*=\s*1",  # OR-based injection
        r'WAITFOR\s+DELAY',  # Time-based injection
        r'BENCHMARK\s*\(',  # Time-based injection
        r'SLEEP\s*\(',  # Time-based injection
        r'INTO\s+OUTFILE',  # File writing
        r'INTO\s+DUMPFILE',  # File writing
        r'LOAD_FILE\s*\(',  # File reading
        r'@@version',  # System variable access
    ]

    # Allowlist for common safe SQL operations in Snowflake
    ALLOWED_OPERATIONS = {
        'SELECT', 'FROM', 'WHERE', 'GROUP BY', 'HAVING', 'ORDER BY',
        'LIMIT', 'JOIN', 'INNER JOIN', 'LEFT JOIN', 'RIGHT JOIN', 'FULL JOIN',
        'CROSS JOIN', 'ON', 'WITH', 'AS', 'CASE', 'WHEN', 'THEN', 'ELSE', 'END',
        'AND', 'OR', 'NOT', 'IS NULL', 'IS NOT NULL', 'LIKE', 'IN', 'BETWEEN',
        'COUNT', 'SUM', 'AVG', 'MIN', 'MAX', 'DISTINCT', 'CAST', 'CONVERT',
        'TO_DATE', 'TO_TIMESTAMP', 'DATE_TRUNC', 'DATEADD', 'DATEDIFF',
        'COALESCE', 'NULLIF', 'IFF', 'IFNULL', 'ZEROIFNULL', 'NVL',
        'FIRST_VALUE', 'LAST_VALUE', 'LAG', 'LEAD', 'RANK', 'DENSE_RANK',
        'ROW_NUMBER', 'NTILE', 'PERCENTILE', 'MEDIAN', 'VAR', 'VARIANCE', 'STDDEV',
        'REGEXP_LIKE', 'REGEXP_REPLACE', 'REGEXP_SUBSTR', 'REGEXP_COUNT',
        'TRIM', 'LTRIM', 'RTRIM', 'UPPER', 'LOWER', 'INITCAP', 'CONCAT', 'SUBSTR',
        'REPLACE', 'POSITION', 'LENGTH', 'CHARINDEX', 'ROUND', 'FLOOR', 'CEIL',
        'ABS', 'MOD', 'POWER', 'SQRT', 'EXP', 'LOG', 'LN', 'GREATEST', 'LEAST'
    }

    @classmethod
    def validate_query(cls, query: str,
                       allow_operations: Optional[Set[str]] = None,
                       max_query_length: int = 10000) -> Tuple[bool, str]:
        """
        Validate a SQL query for safety and permissible operations.

        Args:
            query: SQL query to validate
            allow_operations: Set of additional operations to allow
            max_query_length: Maximum allowed query length

        Returns:
            Tuple[bool, str]: (is_valid, error_message)
        """
        if not query or not query.strip():
            return False, "Query cannot be empty"

        # Check query length
        if len(query) > max_query_length:
            return False, f"Query exceeds maximum length of {max_query_length} characters"

        # Normalize query for consistent validation
        normalized_query = cls._normalize_query(query)

        # Create allowed operations set
        allowed_ops = cls.ALLOWED_OPERATIONS.copy()
        if allow_operations:
            allowed_ops.update(allow_operations)

        # Check for dangerous operations
        for operation in cls.DANGEROUS_OPERATIONS:
            pattern = r'\b' + operation + r'\b'
            if re.search(pattern, normalized_query, re.IGNORECASE):
                # Check if this operation is explicitly allowed
                if operation in allowed_ops:
                    continue
                return False, f"Dangerous operation detected: {operation}"

        # Check for SQL injection patterns
        for pattern in cls.INJECTION_PATTERNS:
            if re.search(pattern, normalized_query, re.IGNORECASE):
                return False, "Potential SQL injection attempt detected"

        return True, "Query validated successfully"

    @classmethod
    def _normalize_query(cls, query: str) -> str:
        """
        Normalize a SQL query by removing comments and extra whitespace.

        Args:
            query: SQL query to normalize

        Returns:
            str: Normalized query
        """
        # Remove SQL comments
        for pattern in cls.COMMENT_PATTERNS:
            query = re.sub(pattern, ' ', query, flags=re.DOTALL)

        # Normalize whitespace
        query = re.sub(r'\s+', ' ', query).strip()

        return query

    @classmethod
    def sanitize_identifiers(cls, identifiers: Union[str, List[str]]) -> Union[str, List[str]]:
        """
        Sanitize table, column, or schema identifiers to prevent injection.

        Args:
            identifiers: String or list of strings to sanitize

        Returns:
            Union[str, List[str]]: Sanitized identifier(s)
        """
        if isinstance(identifiers, list):
            return [cls.sanitize_identifiers(id_) for id_ in identifiers]

        # Allow only alphanumeric and underscore
        sanitized = re.sub(r'[^\w]', '', str(identifiers))

        # Ensure doesn't start with a number
        if sanitized and sanitized[0].isdigit():
            sanitized = 'i_' + sanitized

        return sanitized

    @classmethod
    def build_safe_query(cls,
                         select_columns: List[str],
                         from_table: str,
                         where_conditions: Optional[List[str]] = None,
                         group_by: Optional[List[str]] = None,
                         order_by: Optional[List[str]] = None,
                         limit: Optional[int] = None) -> str:
        """
        Build a safe SQL query from components rather than raw user input.

        Args:
            select_columns: List of columns to select
            from_table: Table to query from
            where_conditions: List of WHERE conditions (will be ANDed together)
            group_by: List of GROUP BY columns
            order_by: List of ORDER BY expressions
            limit: Row limit

        Returns:
            str: Safely constructed SQL query
        """
        # Sanitize inputs
        safe_columns = [cls.sanitize_identifiers(col) for col in select_columns]
        safe_table = cls.sanitize_identifiers(from_table)

        # Build query
        query = f"SELECT {', '.join(safe_columns)} FROM {safe_table}"

        # Add WHERE clause if conditions exist
        if where_conditions and len(where_conditions) > 0:
            safe_conditions = []
            for condition in where_conditions:
                # This is simplified - in real code, you'd need parameter binding
                # instead of string substitution for values
                if cls.validate_query(f"SELECT 1 WHERE {condition}")[0]:
                    safe_conditions.append(condition)
            if safe_conditions:
                query += f" WHERE {' AND '.join(safe_conditions)}"

        # Add GROUP BY clause
        if group_by and len(group_by) > 0:
            safe_group_by = [cls.sanitize_identifiers(col) for col in group_by]
            query += f" GROUP BY {', '.join(safe_group_by)}"

        # Add ORDER BY clause
        if order_by and len(order_by) > 0:
            safe_order_by = []
            for expr in order_by:
                # Parse "column ASC/DESC" format
                parts = expr.split()
                col = cls.sanitize_identifiers(parts[0])
                direction = " ASC" if len(parts) == 1 else f" {parts[1].upper()}"
                if direction.strip() not in [" ASC", " DESC"]:
                    direction = " ASC"
                safe_order_by.append(f"{col}{direction}")
            query += f" ORDER BY {', '.join(safe_order_by)}"

        # Add LIMIT clause
        if limit is not None and isinstance(limit, int) and limit > 0:
            query += f" LIMIT {limit}"

        return query

    @classmethod
    def is_read_only(cls, query: str) -> bool:
        """
        Check if a SQL query is read-only (SELECT only).

        Args:
            query: SQL query to check

        Returns:
            bool: True if query is read-only, False otherwise
        """
        normalized_query = cls._normalize_query(query)

        # Check if query starts with SELECT
        if not re.match(r'^\s*SELECT\s', normalized_query, re.IGNORECASE):
            return False

        # Check for write operations
        for operation in cls.DANGEROUS_OPERATIONS:
            pattern = r'\b' + operation + r'\b'
            if re.search(pattern, normalized_query, re.IGNORECASE):
                return False

        return True


class RequestValidator:
    """
    Validator for structured requests and parameters.
    Ensures inputs conform to expected schemas and are safe to process.
    """

    @staticmethod
    def validate_json_structure(json_str: str) -> Tuple[bool, Union[Dict, str]]:
        """
        Validate that a string contains valid JSON and matches expected structure.

        Args:
            json_str: JSON string to validate

        Returns:
            Tuple[bool, Union[Dict, str]]: (is_valid, parsed_json or error_message)
        """
        if not json_str or not isinstance(json_str, str):
            return False, "Input must be a non-empty string"

        try:
            parsed = json.loads(json_str)
            return True, parsed
        except json.JSONDecodeError as e:
            return False, f"Invalid JSON: {str(e)}"

    @staticmethod
    def validate_schema(data: Dict, schema: Dict) -> Tuple[bool, Optional[str]]:
        """
        Validate that a data structure conforms to expected schema.
        A simplified schema validator (for complex validation, use jsonschema library).

        Args:
            data: Data to validate
            schema: Schema definition with required fields and types

        Returns:
            Tuple[bool, Optional[str]]: (is_valid, error_message)
        """
        # Check required fields
        for field, field_schema in schema.items():
            if field_schema.get('required', False) and field not in data:
                return False, f"Missing required field: {field}"

            # Skip validation if field is not present
            if field not in data:
                continue

            # Validate type
            expected_type = field_schema.get('type')
            if expected_type:
                value = data[field]

                # Check if type matches
                if expected_type == 'string' and not isinstance(value, str):
                    return False, f"Field {field} must be a string"
                elif expected_type == 'number' and not isinstance(value, (int, float)):
                    return False, f"Field {field} must be a number"
                elif expected_type == 'integer' and not isinstance(value, int):
                    return False, f"Field {field} must be an integer"
                elif expected_type == 'boolean' and not isinstance(value, bool):
                    return False, f"Field {field} must be a boolean"
                elif expected_type == 'array' and not isinstance(value, list):
                    return False, f"Field {field} must be an array"
                elif expected_type == 'object' and not isinstance(value, dict):
                    return False, f"Field {field} must be an object"

            # Validate enum
            enum_values = field_schema.get('enum')
            if enum_values and data[field] not in enum_values:
                return False, f"Field {field} must be one of: {', '.join(map(str, enum_values))}"

            # Validate pattern
            pattern = field_schema.get('pattern')
            if pattern and isinstance(data[field], str):
                if not re.match(pattern, data[field]):
                    return False, f"Field {field} must match pattern: {pattern}"

        return True, None

    @staticmethod
    def sanitize_input(input_value: str, max_length: int = 1000) -> str:
        """
        Sanitize a string input to prevent script injection and other attacks.

        Args:
            input_value: String value to sanitize
            max_length: Maximum allowed length

        Returns:
            str: Sanitized string
        """
        if not input_value or not isinstance(input_value, str):
            return ""

        # Truncate to maximum length
        input_value = input_value[:max_length]

        # Remove potentially dangerous HTML/script tags
        input_value = re.sub(r'<script.*?>.*?</script>', '', input_value, flags=re.DOTALL | re.IGNORECASE)
        input_value = re.sub(r'<.*?>', '', input_value)

        # Remove control characters and null bytes
        input_value = re.sub(r'[\x00-\x1F\x7F]', '', input_value)

        return input_value.strip()

    @staticmethod
    def validate_date_format(date_str: str, format_type: str = 'iso') -> bool:
        """
        Validate that a string follows a specific date format.

        Args:
            date_str: Date string to validate
            format_type: Format type ('iso', 'us', 'eu')

        Returns:
            bool: True if date format is valid, False otherwise
        """
        if not date_str or not isinstance(date_str, str):
            return False

        if format_type == 'iso':
            # ISO format (YYYY-MM-DD)
            pattern = r'^\d{4}-\d{2}-\d{2}$'
        elif format_type == 'us':
            # US format (MM/DD/YYYY)
            pattern = r'^\d{2}/\d{2}/\d{4}$'
        elif format_type == 'eu':
            # European format (DD/MM/YYYY)
            pattern = r'^\d{2}/\d{2}/\d{4}$'
        else:
            return False

        return bool(re.match(pattern, date_str))

    @staticmethod
    def validate_email(email: str) -> bool:
        """
        Validate that a string is a properly formatted email address.

        Args:
            email: Email address to validate

        Returns:
            bool: True if email format is valid, False otherwise
        """
        if not email or not isinstance(email, str):
            return False

        # Basic email validation pattern
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, email))

    @staticmethod
    def validate_snowflake_parameters(params: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Validate Snowflake connection parameters.

        Args:
            params: Dictionary of connection parameters

        Returns:
            Tuple[bool, Optional[str]]: (is_valid, error_message)
        """
        required_params = ['account', 'user']

        # Check required parameters
        for param in required_params:
            if param not in params or not params[param]:
                return False, f"Missing required Snowflake parameter: {param}"

        # Either password or private_key_path should be provided
        if ('password' not in params or not params['password']) and \
                ('private_key_path' not in params or not params['private_key_path']):
            return False, "Either password or private_key_path must be provided"

        return True, None

    @staticmethod
    def validate_search_request(query: str) -> Tuple[bool, Optional[str]]:
        """
        Validate a search request string.

        Args:
            query: Search query to validate

        Returns:
            Tuple[bool, Optional[str]]: (is_valid, error_message)
        """
        if not query or not isinstance(query, str):
            return False, "Search query cannot be empty"

        # Check length
        if len(query) < 2:
            return False, "Search query too short"
        if len(query) > 1000:
            return False, "Search query too long (max 1000 characters)"

        # Remove potentially dangerous characters
        sanitized = RequestValidator.sanitize_input(query)
        if sanitized != query:
            return False, "Search query contains potentially unsafe characters"

        return True, None


# Export classes and helper functions
__all__ = ['SQLQueryValidator', 'RequestValidator']


# Example usage
def is_safe_query(query_str: str) -> Tuple[bool, str]:
    """
    Helper function to check if a query is safe to execute.

    Args:
        query_str: SQL query to validate

    Returns:
        Tuple[bool, str]: (is_safe, error_message)
    """
    return SQLQueryValidator.validate_query(query_str)


def sanitize_and_validate_request(request_json: str, schema: Dict) -> Tuple[bool, Union[Dict, str]]:
    """
    Helper function to sanitize and validate a JSON request.

    Args:
        request_json: JSON request string
        schema: Schema definition for validation

    Returns:
        Tuple[bool, Union[Dict, str]]: (is_valid, parsed_request or error_message)
    """
    # Validate JSON structure
    is_valid, result = RequestValidator.validate_json_structure(request_json)
    if not is_valid:
        return False, result

    # Validate against schema
    is_valid, error = RequestValidator.validate_schema(result, schema)
    if not is_valid:
        return False, error

    return True, result