"""
SnowflakeAnalystAgent for translating business questions into SQL queries.
Specializes in database schema understanding and SQL query generation.
"""

import logging
import json
from typing import List, Dict, Any, Optional, Union

from crewai import Task
from pydantic import BaseModel, Field

from agents.base_agent import BaseAgent
from tools.snowflake_tools import SnowflakeQueryTool, get_snowflake_tools
from utils.prompt_templates import SnowflakePromptTemplates
from utils.data_processors import SnowflakeDataProcessor

# Configure logger
logger = logging.getLogger(__name__)


class SnowflakeAnalystAgent(BaseAgent):
    """
    Agent specialized in translating business questions into SQL queries
    and providing business-friendly interpretations of the results.
    """

    def __init__(
            self,
            name: str = "Snowflake Analyst",
            role: str = "Data Analyst and SQL Expert",
            goal: str = "Translate business questions into optimized SQL queries and explain results in business terms",
            backstory: str = None,
            schema_cache: Dict[str, Any] = None,
            **kwargs
    ):
        """
        Initialize a Snowflake Analyst Agent.

        Args:
            name: Agent's name
            role: Agent's role description
            goal: Agent's main objective
            backstory: Agent's background story (optional)
            schema_cache: Cached database schema information
            **kwargs: Additional agent parameters
        """
        # Generate detailed backstory if not provided
        if backstory is None:
            backstory = self._generate_analyst_backstory()

        # Initialize base agent
        super().__init__(
            name=name,
            role=role,
            goal=goal,
            backstory=backstory,
            **kwargs
        )

        # Add Snowflake tools
        self.add_tools(get_snowflake_tools())

        # Initialize schema cache for database metadata
        self.schema_cache = schema_cache or {}

        # Track last query and results for reference
        self.last_query = None
        self.last_results = None
        self.last_explanation = None

        logger.info(f"Initialized {self.__class__.__name__}: {self.name}")

    def _generate_analyst_backstory(self) -> str:
        """
        Generate a detailed backstory for a Snowflake analyst.

        Returns:
            str: Detailed backstory
        """
        return (
            "I am an expert data analyst with deep knowledge of SQL and Snowflake. "
            "I've spent years analyzing business data across finance, sales, marketing, "
            "and operations. My specialty is translating complex business questions into "
            "precise SQL queries that extract meaningful insights. "
            "I excel at understanding database schemas, optimizing query performance, "
            "and explaining technical results in business-friendly terms. "
            "I take pride in my ability to bridge the gap between technical data structures "
            "and business needs, ensuring that decision-makers get accurate, actionable insights."
        )

    async def analyze_business_question(self, question: str) -> Dict[str, Any]:
        """
        Analyze a business question and return SQL and results.

        Args:
            question: Business question in natural language

        Returns:
            Dict[str, Any]: Analysis results including SQL, data, and explanation
        """
        # Step 1: Ensure we have schema information
        await self._ensure_schema_loaded()

        # Step 2: Generate SQL from business question
        sql_query = await self._generate_sql_query(question)
        self.last_query = sql_query

        # Step 3: Execute the SQL query
        query_tool = SnowflakeQueryTool()
        results = query_tool.execute_query(sql_query)
        self.last_results = results

        # Step 4: Generate business explanation of results
        explanation = await self._explain_results(question, sql_query, results)
        self.last_explanation = explanation

        # Return complete analysis
        return {
            "question": question,
            "sql_query": sql_query,
            "results": results,
            "explanation": explanation
        }

    async def _ensure_schema_loaded(self) -> None:
        """
        Ensure that database schema information is loaded into cache.
        """
        if not self.schema_cache:
            logger.info("Loading database schema information...")
            query_tool = SnowflakeQueryTool()

            # Get list of tables
            tables_result = query_tool.list_tables({})

            # Extract table names
            import pandas as pd
            if isinstance(tables_result, str):
                # Parse the formatted string into a DataFrame
                try:
                    # Multi-line string to lines
                    lines = tables_result.strip().split('\n')
                    # Skip header lines
                    data_lines = [line for line in lines if not line.startswith('-') and line.strip()]
                    if len(data_lines) > 1:  # First line is header
                        headers = data_lines[0].split()
                        table_data = []
                        for line in data_lines[1:]:
                            values = line.split()
                            if len(values) >= 1:
                                table_data.append(values[0])  # Get table name

                        self.schema_cache["tables"] = table_data
                except Exception as e:
                    logger.error(f"Error parsing table list: {str(e)}")
                    self.schema_cache["tables"] = []

            # Get schema for each table
            for table_name in self.schema_cache.get("tables", []):
                try:
                    schema_result = query_tool.get_table_schema({"table_name": table_name})
                    self.schema_cache[table_name] = schema_result
                except Exception as e:
                    logger.error(f"Error getting schema for table {table_name}: {str(e)}")

            logger.info(f"Loaded schema for {len(self.schema_cache.get('tables', []))} tables")

    async def _generate_sql_query(self, question: str) -> str:
        """
        Generate optimized SQL query from business question.

        Args:
            question: Business question in natural language

        Returns:
            str: SQL query
        """
        # Prepare schema information for the prompt
        table_schemas = ""
        for table_name in self.schema_cache.get("tables", []):
            table_schemas += f"Table: {table_name}\n"
            table_schemas += f"Schema: {self.schema_cache.get(table_name, 'Schema not available')}\n\n"

        # Generate prompt using template
        prompt = SnowflakePromptTemplates.QUERY_GENERATION.substitute(
            objective=question,
            table_schemas=table_schemas
        )

        # Call LLM to generate SQL
        self.thinking(f"Generating SQL query for: {question}")
        response = await self._llm_client.generate(
            prompt=prompt,
            temperature=0.2,  # Lower temperature for more precise SQL generation
            max_tokens=1000
        )

        # Extract SQL from response
        response_text = response['choices'][0]['message']['content']

        # Try to extract SQL code block if present
        import re
        sql_match = re.search(r'```sql\n(.*?)```', response_text, re.DOTALL)
        if sql_match:
            sql_query = sql_match.group(1).strip()
        else:
            # If no code block, look for lines that look like SQL
            lines = response_text.split('\n')
            sql_lines = []
            for line in lines:
                if any(keyword in line.upper() for keyword in
                       ['SELECT', 'FROM', 'WHERE', 'GROUP BY', 'ORDER BY', 'LIMIT']):
                    sql_lines.append(line)
            sql_query = '\n'.join(sql_lines) if sql_lines else response_text

        return sql_query

    async def _explain_results(self, question: str, sql_query: str, results: str) -> str:
        """
        Generate business-friendly explanation of query results.

        Args:
            question: Original business question
            sql_query: Executed SQL query
            results: Query results

        Returns:
            str: Business-friendly explanation
        """
        prompt = f"""
        I've analyzed a business question by translating it to SQL and running it on our database.

        Original question: "{question}"

        SQL query used:
        ```sql
        {sql_query}
        ```

        Query results:
        ```
        {results}
        ```

        Please provide a clear, business-friendly explanation of these results that addresses the original question.
        Your explanation should:
        1. Summarize the key findings
        2. Highlight any important patterns or trends
        3. Suggest possible business implications
        4. Be concise and avoid technical jargon
        5. Use bullet points for clarity where appropriate
        """

        # Call LLM to generate explanation
        self.thinking(f"Generating explanation for results of question: {question}")
        response = await self._llm_client.generate(
            prompt=prompt,
            temperature=0.5,
            max_tokens=1000
        )

        return response['choices'][0]['message']['content']

    async def suggest_follow_up_questions(self, question: str, results: str) -> List[str]:
        """
        Suggest relevant follow-up questions based on results.

        Args:
            question: Original business question
            results: Query results

        Returns:
            List[str]: List of suggested follow-up questions
        """
        prompt = f"""
        Based on this business question and the analysis results, 
        suggest 3-5 specific follow-up questions that would provide 
        additional valuable insights:

        Original question: "{question}"

        Analysis results:
        ```
        {results}
        ```

        Provide just the follow-up questions, one per line.
        """

        # Call LLM to generate follow-up questions
        response = await self._llm_client.generate(
            prompt=prompt,
            temperature=0.7,
            max_tokens=500
        )

        # Parse response into list of questions
        questions_text = response['choices'][0]['message']['content']
        questions = [q.strip() for q in questions_text.split('\n') if q.strip()]

        # Filter out any non-questions
        questions = [q for q in questions if q.endswith('?')]

        return questions

    async def explain_sql_query(self, sql_query: str) -> str:
        """
        Explain a SQL query in simple business terms.

        Args:
            sql_query: SQL query to explain

        Returns:
            str: Simple explanation of the query
        """
        # Use the query explanation template
        prompt = SnowflakePromptTemplates.QUERY_EXPLANATION.substitute(
            query=sql_query
        )

        # Call LLM to generate explanation
        response = await self._llm_client.generate(
            prompt=prompt,
            temperature=0.3,
            max_tokens=500
        )

        return response['choices'][0]['message']['content']

    async def optimize_sql_query(self, sql_query: str) -> str:
        """
        Optimize a SQL query for better performance.

        Args:
            sql_query: SQL query to optimize

        Returns:
            str: Optimized SQL query
        """
        prompt = f"""
        Please optimize this Snowflake SQL query for better performance:

        ```sql
        {sql_query}
        ```

        Consider:
        1. Appropriate filtering to reduce data scanned
        2. Efficient join strategies
        3. Useful materialized views or cached results
        4. Query structure and readability

        Return only the optimized SQL query.
        """

        # Call LLM to generate optimized query
        response = await self._llm_client.generate(
            prompt=prompt,
            temperature=0.2,
            max_tokens=1000
        )

        optimized_query = response['choices'][0]['message']['content']

        # Extract SQL code block if present
        import re
        sql_match = re.search(r'```sql\n(.*?)```', optimized_query, re.DOTALL)
        if sql_match:
            optimized_query = sql_match.group(1).strip()

        return optimized_query


# Factory function to create an analyst agent with default configuration
def create_snowflake_analyst_agent(**kwargs) -> SnowflakeAnalystAgent:
    """
    Create a Snowflake Analyst Agent with default configuration.

    Args:
        **kwargs: Override default configuration parameters

    Returns:
        SnowflakeAnalystAgent: Configured agent instance
    """
    return SnowflakeAnalystAgent(**kwargs)