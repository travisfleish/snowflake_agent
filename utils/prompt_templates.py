"""
Prompt templates for the Snowflake Agent project.
Contains templates for different agents, tasks, and tools interactions.
"""

from string import Template


class BasePromptTemplates:
    """Base class containing common prompt templates."""

    SYSTEM_BASE = """You are an AI assistant specialized in data analysis with Snowflake.
    Your goal is to help users query, analyze, and visualize data efficiently and accurately."""

    THINKING_TEMPLATE = """Think step by step about how to ${task_description}.
    Consider the available data sources, the necessary operations, and the expected output format."""

    ERROR_HANDLING = """If you encounter an error, do the following:
    1. Identify the root cause of the error
    2. Suggest possible solutions
    3. Implement the most reasonable solution
    4. Verify that the solution resolves the error"""


class SnowflakePromptTemplates:
    """Prompt templates for Snowflake database interactions."""

    QUERY_GENERATION = Template("""
    Please generate a Snowflake SQL query to ${objective}.

    Available tables and their schemas:
    ${table_schemas}

    Consider these performance best practices:
    1. Use appropriate filters to limit data scanned
    2. Select only the necessary columns
    3. Consider using window functions for complex aggregations
    4. Use appropriate join strategies

    The query should be well-formatted, include comments, and be optimized for performance.
    """)

    QUERY_EXPLANATION = Template("""
    Please explain the following Snowflake SQL query in simple terms:

    ```sql
    ${query}
    ```

    Your explanation should cover:
    1. What data is being retrieved
    2. How the data is being filtered and transformed
    3. What the expected output format will be
    4. Any performance considerations
    """)

    DATA_ANALYSIS = Template("""
    Please analyze the following Snowflake query results:

    ```
    ${query_results}
    ```

    Provide the following insights:
    1. Summary statistics of key numerical columns
    2. Notable patterns or trends
    3. Anomalies or outliers
    4. Actionable insights based on the data
    5. Suggested follow-up analyses or queries
    """)

    SCHEMA_EXPLORATION = Template("""
    Please explore the schema of the Snowflake database with the following context:

    Database: ${database}
    Schema: ${schema}

    For each table, identify:
    1. Primary keys and unique identifiers
    2. Foreign key relationships
    3. Important date/time columns for time-series analysis
    4. Numerical columns appropriate for aggregation
    5. Categorical columns useful for grouping

    Suggest potential join paths between related tables.
    """)


class AgentPromptTemplates:
    """Prompt templates specific to different agent roles."""

    ANALYST_AGENT = Template("""
    You are a Data Analyst specialized in Snowflake analytics. 
    Your task is to ${task_description}.

    Available data sources:
    ${data_sources}

    Required analysis techniques:
    ${analysis_techniques}

    Output format:
    ${output_format}

    First, formulate your analytical approach, then implement the necessary queries,
    and finally present your findings in a clear, actionable format.
    """)

    RESEARCH_AGENT = Template("""
    You are a Data Research specialist with expertise in Snowflake.
    Your task is to ${task_description}.

    Areas to investigate:
    ${research_areas}

    Context information:
    ${context}

    Expected deliverables:
    ${deliverables}

    Begin by gathering relevant information, then organize your findings,
    and conclude with a comprehensive research report that answers the key questions.
    """)

    EXECUTIVE_AGENT = Template("""
    You are an Executive Decision Support agent with expertise in data-driven decision making.
    Your task is to ${task_description}.

    Key metrics to consider:
    ${key_metrics}

    Business context:
    ${business_context}

    Decision criteria:
    ${decision_criteria}

    Synthesize the analytical findings into clear, business-focused recommendations.
    Highlight potential risks, opportunities, and trade-offs in your decision support.
    """)


class ToolPromptTemplates:
    """Prompt templates for tool interactions."""

    SNOWFLAKE_CONNECTION = Template("""
    Connect to the Snowflake database with the following parameters:

    Account: ${account}
    Database: ${database}
    Schema: ${schema}
    Warehouse: ${warehouse}
    Role: ${role}

    Verify the connection by executing a test query.
    """)

    DATA_VISUALIZATION = Template("""
    Create a visualization for the following data:

    ```
    ${data}
    ```

    Visualization type: ${viz_type}
    Title: ${title}
    X-axis: ${x_axis}
    Y-axis: ${y_axis}

    Additional configurations:
    ${additional_configs}

    Ensure the visualization is clear, properly labeled, and effectively communicates the key insights.
    """)

    FILE_EXPORT = Template("""
    Export the following data to a ${file_format} file:

    ```
    ${data}
    ```

    File path: ${file_path}
    Include headers: ${include_headers}

    Ensure proper error handling and verify the file was created successfully.
    """)


class CrewPromptTemplates:
    """Prompt templates for crew coordination."""

    TASK_DELEGATION = Template("""
    As the coordinator for this data analysis project, delegate the following task:

    Task: ${task_description}

    Available agents:
    ${available_agents}

    Assignment criteria:
    ${assignment_criteria}

    Provide clear instructions to the assigned agent and specify the expected deliverables.
    """)

    RESULT_INTEGRATION = Template("""
    Integrate the following results from multiple agents:

    Agent 1 (${agent1_name}) results:
    ${agent1_results}

    Agent 2 (${agent2_name}) results:
    ${agent2_results}

    Agent 3 (${agent3_name}) results:
    ${agent3_results}

    Create a coherent final output that combines the insights from all agents,
    resolves any conflicts, and presents a unified analysis.
    """)


# Export all template classes
__all__ = [
    'BasePromptTemplates',
    'SnowflakePromptTemplates',
    'AgentPromptTemplates',
    'ToolPromptTemplates',
    'CrewPromptTemplates'
]