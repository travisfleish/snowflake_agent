"""
BusinessResearchCrew for answering ad hoc business questions using Snowflake data.
Specializes in research, data analysis, and providing insights for business decisions.
"""

import logging
import datetime
import re
from typing import List, Dict, Any, Optional, Union, Tuple

from crewai import Task
from pydantic import BaseModel, Field

from crews.base_crew import BaseCrew, CrewConfig
from agents.research_agent import SnowflakeAnalystAgent
from agents.analyst_agent import DataInsightAgent
from agents.executive_agent import StrategicAdvisorAgent
from tools.snowflake_tools import get_snowflake_tools
from utils.data_processors import SnowflakeDataProcessor
from utils.prompt_templates import SnowflakePromptTemplates

# Configure logger
logger = logging.getLogger(__name__)


class ResearchQuestion(BaseModel):
    """Model for a business research question."""

    question: str = Field(..., description="The main business question to research")
    context: Optional[str] = Field(None, description="Additional context for the question")
    required_data_sources: List[str] = Field(default=[], description="Specific data sources required")
    time_frame: Optional[str] = Field(None, description="Time frame for the analysis")
    priority: str = Field(default="medium", description="Priority level (low, medium, high)")
    requested_by: Optional[str] = Field(None, description="Person/department requesting the research")
    output_format: str = Field(default="report",
                               description="Preferred output format (report, dashboard, presentation)")


class BusinessResearchCrew(BaseCrew):
    """
    Specialized crew for answering ad hoc business questions using Snowflake data.
    Performs research, analyzes data, and generates insights for business decisions.
    """

    def __init__(
            self,
            name: str = "Business Research Crew",
            description: str = "Research and answer business questions using Snowflake data",
            database_schema_cache: Dict[str, Any] = None,
            data_dictionary: Dict[str, Any] = None,
            business_context: Dict[str, Any] = None,
            sequential: bool = True,
            **kwargs
    ):
        """
        Initialize a Business Research Crew.

        Args:
            name: Crew name
            description: Crew description
            database_schema_cache: Cached database schema information
            data_dictionary: Data dictionary with field descriptions
            business_context: Business context for the organization
            sequential: Whether to run tasks sequentially
            **kwargs: Additional crew parameters
        """
        # Create crew config
        config = CrewConfig(
            name=name,
            description=description,
            sequential=sequential,
            **kwargs
        )

        # Initialize base crew
        super().__init__(config=config)

        # Set database metadata
        self.database_schema_cache = database_schema_cache or {}
        self.data_dictionary = data_dictionary or {}

        # Set business context
        self.business_context = business_context or {
            "company_description": "Mid-sized retail company with online and physical stores",
            "primary_business_areas": ["retail", "e-commerce"],
            "key_metrics": ["sales", "margin", "customer_retention", "inventory_turnover"],
            "fiscal_year": "January to December",
            "reporting_periods": "Quarterly and Monthly",
            "key_stakeholders": ["Executive Team", "Sales", "Marketing", "Operations", "Finance"]
        }

        # Initialize research history
        self.research_history = []

        # Initialize default agents if none are provided
        if not self.agents:
            self._initialize_default_agents()

    def _initialize_default_agents(self) -> None:
        """
        Initialize default agents for the business research crew.
        """
        # Create a data engineer agent for query generation and execution
        data_engineer = SnowflakeAnalystAgent(
            name="Data Engineer",
            role="SQL Expert & Data Engineer",
            goal="Generate optimal Snowflake queries and retrieve accurate data",
            backstory=(
                "I am a skilled data engineer with expertise in Snowflake. "
                "I translate complex business questions into efficient SQL queries "
                "and ensure data is accurately retrieved and properly prepared for analysis."
            ),
            schema_cache=self.database_schema_cache,
            tools=get_snowflake_tools()
        )

        # Create a business analyst agent for data analysis
        business_analyst = DataInsightAgent(
            name="Business Analyst",
            role="Data Analyst & Business Intelligence Specialist",
            goal="Analyze data to generate relevant business insights",
            backstory=(
                "I specialize in transforming raw data into meaningful business insights. "
                "My expertise lies in spotting trends, identifying anomalies, and extracting "
                "actionable information that directly addresses business questions."
            )
        )

        # Create a research director agent for question interpretation and final response
        research_director = StrategicAdvisorAgent(
            name="Research Director",
            role="Business Research Director",
            goal="Interpret business questions and synthesize findings into coherent answers",
            backstory=(
                "I am a seasoned business research director with deep industry knowledge. "
                "I excel at understanding the true intent behind business questions, "
                "directing research efforts, and synthesizing technical findings into "
                "clear, actionable business recommendations."
            ),
            business_context=self.business_context
        )

        # Add agents to the crew
        self.add_agents([data_engineer, business_analyst, research_director])

    def _extract_required_tables(self, question: str) -> List[str]:
        """
        Extract potentially required tables from the question based on schema cache.

        Args:
            question: Business question

        Returns:
            List[str]: List of potentially relevant table names
        """
        tables = []

        # Look for table names in the schema cache that might be relevant
        if self.database_schema_cache.get("tables"):
            for table_name in self.database_schema_cache.get("tables"):
                # Simple approach: if the table name (without underscores) appears in the question
                table_words = table_name.replace("_", " ").lower()
                if table_words in question.lower():
                    tables.append(table_name)

        return tables

    def _extract_data_fields(self, question: str) -> List[str]:
        """
        Extract potentially relevant data fields from the question.

        Args:
            question: Business question

        Returns:
            List[str]: List of potentially relevant field names
        """
        fields = []

        # Extract fields from data dictionary
        for field, info in self.data_dictionary.items():
            field_name = field.replace("_", " ").lower()
            aliases = info.get("aliases", [])
            descriptions = info.get("description", "").lower()

            # Check if field name or aliases appear in the question
            if field_name in question.lower():
                fields.append(field)
            else:
                for alias in aliases:
                    if alias.lower() in question.lower():
                        fields.append(field)
                        break

        return fields

    def _extract_time_frame(self, question: str) -> Optional[str]:
        """
        Extract time frame reference from the question.

        Args:
            question: Business question

        Returns:
            Optional[str]: Extracted time frame or None
        """
        # Common time frame patterns
        time_patterns = [
            r"(last|previous) (year|quarter|month|week|day)",
            r"(ytd|year to date|quarter to date|month to date|qtd|mtd)",
            r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]* \d{4}",
            r"q[1-4] \d{4}",
            r"(20\d{2})-(20\d{2})",
            r"from .{3,20} to .{3,20}",
            r"since [a-z]+ \d{4}"
        ]

        for pattern in time_patterns:
            match = re.search(pattern, question.lower())
            if match:
                return match.group(0)

        return None

    def _parse_data_sources(self, research_question: ResearchQuestion) -> List[Dict[str, Any]]:
        """
        Parse and identify required data sources for the question.

        Args:
            research_question: Research question details

        Returns:
            List[Dict[str, Any]]: Data sources information
        """
        question_text = research_question.question
        if research_question.context:
            question_text += " " + research_question.context

        # Extract explicitly mentioned data sources
        explicit_sources = research_question.required_data_sources

        # Add implicitly mentioned tables
        implicit_tables = self._extract_required_tables(question_text)

        # Add relevant fields
        relevant_fields = self._extract_data_fields(question_text)

        # Extract time frame
        time_frame = research_question.time_frame or self._extract_time_frame(question_text)

        # Combine all sources
        data_sources = []

        # Add explicit sources
        for source in explicit_sources:
            data_sources.append({
                "name": source,
                "type": "explicit",
                "relevance": "high"
            })

        # Add implicit tables
        for table in implicit_tables:
            if table not in explicit_sources:
                data_sources.append({
                    "name": table,
                    "type": "table",
                    "relevance": "medium"
                })

        # Record time frame
        if time_frame:
            data_sources.append({
                "name": "time_frame",
                "value": time_frame,
                "type": "filter",
                "relevance": "high"
            })

        # Record fields
        if relevant_fields:
            data_sources.append({
                "name": "fields",
                "value": relevant_fields,
                "type": "fields",
                "relevance": "medium"
            })

        return data_sources

    def setup_research_workflow(self, research_question: ResearchQuestion) -> None:
        """
        Set up the research workflow tasks based on the question.

        Args:
            research_question: Research question details
        """
        # Get the agents
        data_engineer = next((a for a in self.agents if isinstance(a, SnowflakeAnalystAgent)), None)
        business_analyst = next((a for a in self.agents if isinstance(a, DataInsightAgent)), None)
        research_director = next((a for a in self.agents if isinstance(a, StrategicAdvisorAgent)), None)

        if not data_engineer or not business_analyst or not research_director:
            raise ValueError("Research crew requires all three agent types")

        # Parse data sources
        data_sources = self._parse_data_sources(research_question)
        data_sources_text = "\n".join([f"- {source['name']} (Type: {source['type']}, Relevance: {source['relevance']})"
                                       for source in data_sources])

        # Create question interpretation task for the director
        interpret_task = self.create_task(
            description=f"""
            Interpret the following business research question to determine the best approach:

            Question: "{research_question.question}"
            {f'Context: {research_question.context}' if research_question.context else ''}
            Priority: {research_question.priority}
            Requested by: {research_question.requested_by or 'Unknown'}

            Potential data sources identified:
            {data_sources_text}

            Your task:
            1. Clarify the core business question and what decision(s) it aims to inform
            2. Break down the question into specific analytical components
            3. Identify any additional data sources that might be relevant
            4. Provide guidance on analytical approach and key metrics to consider
            5. Outline what a good answer to this question would include

            Output your interpretation with clear instructions for the data engineer.
            """,
            agent=research_director,
            expected_output="Detailed question interpretation and research plan"
        )

        # Create data retrieval task
        data_task = self.create_task(
            description=f"""
            Based on the research plan provided by the Research Director, retrieve the necessary data from Snowflake:

            Your task:
            1. Review the research plan and question interpretation
            2. Develop SQL queries to extract the required data from Snowflake
            3. Execute the queries and gather all necessary data
            4. Ensure data is complete, accurate, and properly formatted
            5. Document any data limitations, assumptions, or quality issues

            Pay special attention to proper joining of tables, filtering for the correct time period,
            and including all necessary dimensions and metrics.

            Optimize your queries for performance while ensuring accuracy.
            """,
            agent=data_engineer,
            expected_output="Retrieved data sets with documentation"
        )

        # Create analysis task
        analysis_task = self.create_task(
            description=f"""
            Analyze the data provided by the Data Engineer to address the business question:

            Question: "{research_question.question}"
            {f'Context: {research_question.context}' if research_question.context else ''}

            Your task:
            1. Thoroughly analyze the retrieved data sets
            2. Identify key patterns, trends, correlations, and anomalies
            3. Calculate relevant metrics and perform statistical analyses as needed
            4. Create clear visualizations that illustrate key findings
            5. Document your analytical approach and any assumptions made

            Ensure your analysis directly addresses the core business question and provides
            actionable insights rather than just describing the data.
            """,
            agent=business_analyst,
            expected_output="Comprehensive analysis with insights and visualizations"
        )

        # Create synthesis task
        synthesis_task = self.create_task(
            description=f"""
            Synthesize the research findings into a comprehensive answer to the original business question:

            Question: "{research_question.question}"
            {f'Context: {research_question.context}' if research_question.context else ''}
            Requested output format: {research_question.output_format}

            Your task:
            1. Review all data and analysis from the previous steps
            2. Synthesize the findings into a clear, cohesive narrative
            3. Provide direct answers to the business question with supporting evidence
            4. Highlight key insights and their business implications
            5. Suggest concrete recommendations or next steps based on the findings
            6. Format the response appropriately for the requested output format

            Make sure your response is business-friendly, actionable, and focused on
            implications rather than methodology.
            """,
            agent=research_director,
            expected_output=f"Final research answer in {research_question.output_format} format"
        )

        logger.info(f"Set up research workflow for question: {research_question.question}")

    async def answer_research_question(self, question: Union[str, ResearchQuestion]) -> Dict[str, Any]:
        """
        Answer a business research question using Snowflake data.

        Args:
            question: Business question or ResearchQuestion object

        Returns:
            Dict[str, Any]: Research results
        """
        # Convert string to ResearchQuestion if needed
        if isinstance(question, str):
            research_question = ResearchQuestion(
                question=question,
                output_format="report"
            )
        else:
            research_question = question

        # Reset the crew for a new research question
        self.reset()

        # Set up workflow
        self.setup_research_workflow(research_question)

        # Run the crew
        result = await self.run()

        # Process and structure the results
        try:
            # This is a simplified approach to parsing the results
            # In a real implementation, you might have structured outputs

            # Try to parse sections based on headers
            import re

            # Extract question interpretation
            interpretation_match = re.search(
                r'# Question Interpretation(.*?)(?=# Data Retrieval|# Analysis|# Answer|$)',
                result, re.DOTALL)
            interpretation = interpretation_match.group(1).strip() if interpretation_match else None

            # Extract data retrieval information
            data_match = re.search(r'# Data Retrieval(.*?)(?=# Analysis|# Answer|$)', result, re.DOTALL)
            data_retrieval = data_match.group(1).strip() if data_match else None

            # Extract analysis
            analysis_match = re.search(r'# Analysis(.*?)(?=# Answer|$)', result, re.DOTALL)
            analysis = analysis_match.group(1).strip() if analysis_match else None

            # Extract final answer
            answer_match = re.search(r'# Answer(.*?)$', result, re.DOTALL)
            answer = answer_match.group(1).strip() if answer_match else result

            # Generate summary
            summary = await self.generate_execution_summary()

            # Create research record
            research_record = {
                "question": research_question.dict(),
                "interpretation": interpretation,
                "data_retrieval": data_retrieval,
                "analysis": analysis,
                "answer": answer,
                "execution_summary": summary,
                "timestamp": datetime.datetime.now().isoformat()
            }

            # Add to research history
            self.research_history.append(research_record)

            return research_record

        except Exception as e:
            logger.error(f"Error processing research results: {str(e)}")

            # Create minimal research record
            research_record = {
                "question": research_question.dict(),
                "raw_result": result,
                "error": str(e),
                "timestamp": datetime.datetime.now().isoformat()
            }

            # Add to research history
            self.research_history.append(research_record)

            return research_record

    async def get_similar_past_research(self, question: str, max_results: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieve similar research from history based on question similarity.

        Args:
            question: The business question to find similar research for
            max_results: Maximum number of results to return

        Returns:
            List[Dict[str, Any]]: Similar past research
        """
        if not self.research_history:
            return []

        # Get the research director agent
        research_director = next((a for a in self.agents if isinstance(a, StrategicAdvisorAgent)), None)
        if not research_director:
            return []

        # Create a ranking task
        history_questions = [
            f"{i + 1}. {record['question']['question']}"
            for i, record in enumerate(self.research_history)
        ]

        questions_text = "\n".join(history_questions)

        task = Task(
            description=f"""
            Review these past research questions and rank them by similarity to this new question:

            New question: "{question}"

            Past questions:
            {questions_text}

            Rank the top {max_results} most similar questions by their number.
            Explain your reasoning for each selected question.
            Format your response as JSON with structure:
            {{
                "rankings": [
                    {{"number": 1, "similarity": "high", "reasoning": "explanation"}},
                    ...
                ]
            }}
            """,
            agent=research_director.get_crew_agent(),
            expected_output="JSON ranking of similar questions"
        )

        # Execute this task
        from crewai import Crew
        ranking_crew = Crew(
            agents=[research_director.get_crew_agent()],
            tasks=[task],
            verbose=self.config.verbose
        )

        result = await ranking_crew.run_async()

        # Parse results (simplified approach)
        import json
        import re

        json_match = re.search(r'({.*})', result, re.DOTALL)
        if json_match:
            try:
                rankings = json.loads(json_match.group(1))

                # Get the similar research
                similar_research = []
                for rank in rankings.get("rankings", []):
                    question_num = rank.get("number")
                    if question_num and 1 <= question_num <= len(self.research_history):
                        research = self.research_history[question_num - 1]
                        similar_research.append({
                            "question": research["question"]["question"],
                            "answer": research.get("answer", "No answer available"),
                            "timestamp": research["timestamp"],
                            "similarity": rank.get("similarity", "unknown"),
                            "reasoning": rank.get("reasoning", "No reasoning provided")
                        })

                return similar_research[:max_results]

            except:
                logger.error("Error parsing similarity rankings")

        return []

    async def generate_multi_perspective_answer(self, question: str) -> Dict[str, Any]:
        """
        Generate answers from multiple business perspectives.

        Args:
            question: Business question

        Returns:
            Dict[str, Any]: Answers from different perspectives
        """
        # Define different business perspectives
        perspectives = [
            {"name": "Financial", "focus": "Cost implications, ROI, budgeting, financial risk"},
            {"name": "Customer", "focus": "Customer experience, satisfaction, retention, loyalty"},
            {"name": "Operational", "focus": "Efficiency, process improvement, resource utilization"},
            {"name": "Strategic", "focus": "Long-term goals, market positioning, competitive advantage"}
        ]

        # Get the research director agent
        research_director = next((a for a in self.agents if isinstance(a, StrategicAdvisorAgent)), None)
        if not research_director:
            return {"error": "Research director agent not available"}

        # Generate a main answer first
        main_results = await self.answer_research_question(question)
        main_answer = main_results.get("answer", "")

        # For each perspective, create a specialized task
        perspective_answers = {}

        for perspective in perspectives:
            task = Task(
                description=f"""
                Re-interpret the research findings from a {perspective['name']} perspective:

                Original question: "{question}"

                Research findings:
                {main_answer}

                Your task:
                Analyze these findings specifically from a {perspective['name']} perspective, focusing on:
                {perspective['focus']}

                Provide a focused answer (300-500 words) that addresses the original question
                but with specific emphasis on {perspective['name']} considerations.
                """,
                agent=research_director.get_crew_agent(),
                expected_output=f"{perspective['name']} perspective answer"
            )

            # Execute this task
            from crewai import Crew
            perspective_crew = Crew(
                agents=[research_director.get_crew_agent()],
                tasks=[task],
                verbose=self.config.verbose
            )

            result = await perspective_crew.run_async()
            perspective_answers[perspective["name"]] = result

        # Compile all perspectives
        return {
            "question": question,
            "main_answer": main_answer,
            "perspectives": perspective_answers,
            "timestamp": datetime.datetime.now().isoformat()
        }

    def get_research_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get the research history.

        Args:
            limit: Maximum number of history items to return

        Returns:
            List[Dict[str, Any]]: Research history
        """
        if limit is not None:
            return self.research_history[-limit:]
        return self.research_history

    def _extract_queries_from_result(self, result: str) -> List[str]:
        """
        Extract SQL queries from a result string.

        Args:
            result: Result string

        Returns:
            List[str]: Extracted SQL queries
        """
        # Find SQL code blocks
        sql_blocks = re.findall(r'```sql\n(.*?)\n```', result, re.DOTALL)

        # If no code blocks found, look for SQL keywords
        if not sql_blocks:
            potential_queries = []
            lines = result.split('\n')
            current_query = []
            in_query = False

            for line in lines:
                # Check if line contains SQL keywords
                sql_keywords = ['SELECT', 'FROM', 'WHERE', 'GROUP BY', 'ORDER BY', 'HAVING', 'JOIN']
                if any(keyword in line.upper() for keyword in sql_keywords):
                    in_query = True
                    current_query.append(line)
                elif in_query and line.strip() and not line.startswith('#'):
                    current_query.append(line)
                elif in_query and (not line.strip() or line.startswith('#')):
                    if current_query:
                        potential_queries.append('\n'.join(current_query))
                        current_query = []
                        in_query = False

            # Add the last query if there is one
            if current_query:
                potential_queries.append('\n'.join(current_query))

            return potential_queries

        return sql_blocks

    async def export_research_as_document(
            self,
            research_id: int,
            format: str = "markdown"
    ) -> str:
        """
        Export a research record as a formatted document.

        Args:
            research_id: Index of the research record in history
            format: Output format ("markdown", "html", "text")

        Returns:
            str: Formatted document
        """
        if not self.research_history or research_id >= len(self.research_history):
            return f"Error: Research record with ID {research_id} not found"

        research = self.research_history[research_id]

        # Extract core components
        question = research["question"]["question"]
        context = research["question"].get("context", "")

        interpretation = research.get("interpretation", "")
        data_retrieval = research.get("data_retrieval", "")
        analysis = research.get("analysis", "")
        answer = research.get("answer", "")

        # Extract SQL queries if available
        queries = self._extract_queries_from_result(data_retrieval)
        queries_text = "\n\n".join([f"```sql\n{query}\n```" for query in queries])

        # Format based on requested output
        if format == "markdown":
            document = f"""
# Business Research Report

## Question
**{question}**

{f'**Context:** {context}' if context else ''}

## Executive Summary
{answer[:500] + '...' if len(answer) > 500 else answer}

## Detailed Analysis
{analysis}

## Data Sources
{data_retrieval.replace(queries_text, "")}

## SQL Queries Used
{queries_text}

## Methodology
{interpretation}

---
*Research generated on {research["timestamp"]}*
"""

        elif format == "html":
            # Simple HTML formatting
            document = f"""
<html>
<head>
    <title>Business Research Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #2c3e50; }}
        h2 {{ color: #3498db; }}
        pre {{ background-color: #f8f9fa; padding: 10px; border-radius: 5px; }}
        .query {{ background-color: #f0f8ff; padding: 10px; border-left: 3px solid #3498db; }}
        .timestamp {{ color: #7f8c8d; font-style: italic; }}
    </style>
</head>
<body>
    <h1>Business Research Report</h1>

    <h2>Question</h2>
    <p><strong>{question}</strong></p>
    {f'<p><strong>Context:</strong> {context}</p>' if context else ''}

    <h2>Executive Summary</h2>
    <p>{answer[:500] + '...' if len(answer) > 500 else answer}</p>

    <h2>Detailed Analysis</h2>
    <div>{analysis}</div>

    <h2>Data Sources</h2>
    <div>{data_retrieval.replace(queries_text, "")}</div>

    <h2>SQL Queries Used</h2>
    {''.join([f'<pre class="query">{query}</pre>' for query in queries])}

    <h2>Methodology</h2>
    <div>{interpretation}</div>

    <hr>
    <p class="timestamp">Research generated on {research["timestamp"]}</p>
</body>
</html>
"""

        else:  # text format
            document = f"""
BUSINESS RESEARCH REPORT

QUESTION:
{question}

{f'CONTEXT: {context}' if context else ''}

EXECUTIVE SUMMARY:
{answer[:500] + '...' if len(answer) > 500 else answer}

DETAILED ANALYSIS:
{analysis}

DATA SOURCES:
{data_retrieval.replace(queries_text, "")}

SQL QUERIES USED:
{queries_text.replace("```sql", "").replace("```", "")}

METHODOLOGY:
{interpretation}

-------------------------
Research generated on {research["timestamp"]}
"""

        return document


# Factory function to create a business research crew with default configuration
def create_business_research_crew(**kwargs) -> BusinessResearchCrew:
    """
    Create a Business Research Crew with default configuration.

    Args:
        **kwargs: Override default configuration parameters

    Returns:
        BusinessResearchCrew: Configured crew instance
    """
    return BusinessResearchCrew(**kwargs)