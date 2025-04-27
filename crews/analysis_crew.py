"""
SalesAnalysisCrew for analyzing sales data and providing business recommendations.
Fetches data, performs analysis, and generates actionable insights.
"""

import logging
import datetime
from typing import List, Dict, Any, Optional, Union

from crewai import Task
from pydantic import BaseModel, Field

from crews.base_crew import BaseCrew, CrewConfig
from agents.analyst_agent import DataInsightAgent
from agents.executive_agent import StrategicAdvisorAgent
from agents.research_agent import SnowflakeAnalystAgent
from tools.snowflake_tools import get_snowflake_tools
from utils.data_processors import SnowflakeDataProcessor

# Configure logger
logger = logging.getLogger(__name__)


class SalesDataConfig(BaseModel):
    """Configuration for sales data analysis."""

    time_period: str = Field("last_quarter",
                             description="Time period to analyze (last_month, last_quarter, last_year, custom)")
    custom_start_date: Optional[str] = Field(None,
                                             description="Custom start date (YYYY-MM-DD) if time_period is 'custom'")
    custom_end_date: Optional[str] = Field(None, description="Custom end date (YYYY-MM-DD) if time_period is 'custom'")
    product_categories: List[str] = Field(default=["all"], description="Product categories to analyze")
    regions: List[str] = Field(default=["all"], description="Sales regions to analyze")
    metrics: List[str] = Field(
        default=["revenue", "units_sold", "profit_margin", "customer_retention"],
        description="Metrics to include in analysis"
    )
    comparison_period: Optional[str] = Field(None,
                                             description="Period to compare against (previous_period, year_over_year, none)")
    segmentation: List[str] = Field(
        default=["product", "region", "customer_type"],
        description="Dimensions for segmentation analysis"
    )


class SalesAnalysisCrew(BaseCrew):
    """
    Specialized crew for sales data analysis and business recommendations.
    Fetches sales data, performs analysis, and recommends business actions.
    """

    def __init__(
            self,
            name: str = "Sales Analysis Crew",
            description: str = "Analyze sales data and generate business recommendations",
            sales_config: Optional[SalesDataConfig] = None,
            business_context: Dict[str, Any] = None,
            **kwargs
    ):
        """
        Initialize a Sales Analysis Crew.

        Args:
            name: Crew name
            description: Crew description
            sales_config: Configuration for sales data analysis
            business_context: Business context for recommendations
            **kwargs: Additional crew parameters
        """
        # Create crew config
        config = CrewConfig(
            name=name,
            description=description,
            **kwargs
        )

        # Initialize base crew
        super().__init__(config=config)

        # Set sales configuration
        self.sales_config = sales_config or SalesDataConfig()

        # Set business context
        self.business_context = business_context or {
            "company_size": "midsize",
            "industry": "retail",
            "market_position": "growing",
            "strategic_goals": ["increase_market_share", "improve_profitability", "expand_product_lines"],
            "key_challenges": ["increased_competition", "supply_chain_issues", "customer_acquisition_costs"],
            "risk_tolerance": "moderate"
        }

        # Initialize default agents if none are provided
        if not self.agents:
            self._initialize_default_agents()

        # Analysis results storage
        self.data_fetching_results = None
        self.analysis_results = None
        self.recommendation_results = None

    def _initialize_default_agents(self) -> None:
        """
        Initialize default agents for the sales analysis crew.
        """
        # Create a data fetching agent
        data_agent = SnowflakeAnalystAgent(
            name="Sales Data Analyst",
            role="SQL Expert for Sales Data",
            goal="Fetch and prepare comprehensive sales data from Snowflake",
            tools=get_snowflake_tools()
        )

        # Create a data insight agent
        insight_agent = DataInsightAgent(
            name="Sales Insight Analyst",
            role="Analyze sales data for patterns and insights",
            goal="Discover meaningful trends and opportunities in sales data"
        )

        # Create a strategic advisor agent
        advisor_agent = StrategicAdvisorAgent(
            name="Sales Strategy Advisor",
            role="Sales Strategy Consultant",
            goal="Transform sales insights into actionable business recommendations",
            business_context=self.business_context
        )

        # Add agents to the crew
        self.add_agents([data_agent, insight_agent, advisor_agent])

    def _get_date_range_sql(self) -> str:
        """
        Generate SQL date range clause based on sales configuration.

        Returns:
            str: SQL WHERE clause for date filtering
        """
        today = datetime.datetime.now()

        if self.sales_config.time_period == "custom" and self.sales_config.custom_start_date and self.sales_config.custom_end_date:
            return f"transaction_date >= '{self.sales_config.custom_start_date}' AND transaction_date <= '{self.sales_config.custom_end_date}'"

        elif self.sales_config.time_period == "last_month":
            first_day = (today.replace(day=1) - datetime.timedelta(days=1)).replace(day=1)
            last_day = today.replace(day=1) - datetime.timedelta(days=1)
            return f"transaction_date >= '{first_day.strftime('%Y-%m-%d')}' AND transaction_date <= '{last_day.strftime('%Y-%m-%d')}'"

        elif self.sales_config.time_period == "last_quarter":
            # Calculate first day of previous quarter
            month = today.month
            quarter = (month - 1) // 3
            if quarter == 0:
                first_month = 10
                year = today.year - 1
            else:
                first_month = (quarter - 1) * 3 + 1
                year = today.year

            first_day = datetime.datetime(year, first_month, 1)
            last_day = datetime.datetime(year, first_month + 2, 1) + datetime.timedelta(days=31)
            last_day = last_day.replace(day=1) - datetime.timedelta(days=1)

            return f"transaction_date >= '{first_day.strftime('%Y-%m-%d')}' AND transaction_date <= '{last_day.strftime('%Y-%m-%d')}'"

        elif self.sales_config.time_period == "last_year":
            first_day = datetime.datetime(today.year - 1, 1, 1)
            last_day = datetime.datetime(today.year - 1, 12, 31)
            return f"transaction_date >= '{first_day.strftime('%Y-%m-%d')}' AND transaction_date <= '{last_day.strftime('%Y-%m-%d')}'"

        else:
            # Default to last 90 days
            start_date = today - datetime.timedelta(days=90)
            return f"transaction_date >= '{start_date.strftime('%Y-%m-%d')}'"

    def _get_product_region_filter_sql(self) -> str:
        """
        Generate SQL filter clauses for products and regions.

        Returns:
            str: SQL WHERE clause for product and region filtering
        """
        filters = []

        # Product category filter
        if "all" not in self.sales_config.product_categories:
            categories = "', '".join(self.sales_config.product_categories)
            filters.append(f"product_category IN ('{categories}')")

        # Region filter
        if "all" not in self.sales_config.regions:
            regions = "', '".join(self.sales_config.regions)
            filters.append(f"region IN ('{regions}')")

        if filters:
            return " AND " + " AND ".join(filters)
        return ""

    def setup_sales_analysis_workflow(self) -> None:
        """
        Set up the sales analysis workflow tasks.
        """
        # Get the agents
        data_agent = next((a for a in self.agents if isinstance(a, SnowflakeAnalystAgent)), None)
        insight_agent = next((a for a in self.agents if isinstance(a, DataInsightAgent)), None)
        advisor_agent = next((a for a in self.agents if isinstance(a, StrategicAdvisorAgent)), None)

        if not data_agent or not insight_agent or not advisor_agent:
            raise ValueError("Sales analysis crew requires all three agent types")

        # Generate SQL query parameters
        date_range = self._get_date_range_sql()
        filters = self._get_product_region_filter_sql()
        metrics = ", ".join(self.sales_config.metrics)
        segmentation = ", ".join(self.sales_config.segmentation)

        # Create data fetching task
        fetch_task = self.create_task(
            description=f"""
            Fetch sales data from the sales_transactions table with the following requirements:
            - Time period: {date_range}
            - Additional filters: {filters}
            - Include these metrics: {metrics}
            - Include these dimensions for segmentation: {segmentation}

            The query should provide comprehensive sales data that enables detailed analysis.
            If comparison is needed, also fetch data for {self.sales_config.comparison_period or 'a suitable comparison period'}.
            Make sure to optimize the query for performance while ensuring all necessary data is retrieved.
            """,
            agent=data_agent,
            expected_output="Comprehensive sales data from Snowflake"
        )

        # Create analysis task
        analysis_task = self.create_task(
            description=f"""
            Analyze the sales data provided from the previous task to identify:
            1. Overall trends in {', '.join(self.sales_config.metrics)}
            2. Performance across different {', '.join(self.sales_config.segmentation)}
            3. Notable anomalies or outliers in the data
            4. Correlation between different metrics
            5. Comparative analysis with {self.sales_config.comparison_period or 'previous periods'} if available

            Generate visualizations that effectively communicate the key insights.
            Provide a detailed narrative explaining the findings in business terms.
            Focus on actionable insights rather than just describing the data.
            """,
            agent=insight_agent,
            expected_output="Detailed sales analysis with visualizations and insights"
        )

        # Create recommendation task
        recommendation_task = self.create_task(
            description=f"""
            Based on the sales data analysis from the previous task, develop strategic business recommendations:

            Business context:
            - Company size: {self.business_context.get('company_size')}
            - Industry: {self.business_context.get('industry')}
            - Market position: {self.business_context.get('market_position')}
            - Strategic goals: {', '.join(self.business_context.get('strategic_goals', []))}
            - Key challenges: {', '.join(self.business_context.get('key_challenges', []))}
            - Risk tolerance: {self.business_context.get('risk_tolerance')}

            Provide 3-5 specific, actionable recommendations that address:
            1. Optimizing product mix based on performance
            2. Targeting high-value regions or customer segments
            3. Pricing or promotion strategies to improve metrics
            4. Addressing underperforming areas
            5. Capitalizing on identified opportunities

            For each recommendation, include expected impact, implementation timeline, 
            required resources, and metrics to track success.
            """,
            agent=advisor_agent,
            expected_output="Strategic business recommendations with implementation plan"
        )

        logger.info("Set up sales analysis workflow")

    async def execute_sales_analysis(self) -> Dict[str, Any]:
        """
        Execute the full sales analysis workflow.

        Returns:
            Dict[str, Any]: Analysis results and recommendations
        """
        # Set up workflow
        self.setup_sales_analysis_workflow()

        # Run the crew
        result = await self.run()

        # Process and structure the results
        try:
            # This is a simplified approach to parsing the results
            # In a real implementation, you might have structured outputs or better delimitation

            # Try to parse sections based on headers
            import re

            # Extract data fetching results
            data_match = re.search(r'# Data Fetching Results(.*?)(?=# Analysis Results|# Recommendations|$)', result,
                                   re.DOTALL)
            self.data_fetching_results = data_match.group(1).strip() if data_match else None

            # Extract analysis results
            analysis_match = re.search(r'# Analysis Results(.*?)(?=# Recommendations|$)', result, re.DOTALL)
            self.analysis_results = analysis_match.group(1).strip() if analysis_match else None

            # Extract recommendations
            recommendation_match = re.search(r'# Recommendations(.*?)$', result, re.DOTALL)
            self.recommendation_results = recommendation_match.group(1).strip() if recommendation_match else None

            # If no clear sections, use the whole result
            if not self.data_fetching_results and not self.analysis_results and not self.recommendation_results:
                self.analysis_results = result

            # Generate summary
            summary = await self.generate_execution_summary()

            # Create structured response
            analysis_report = {
                "config": {
                    "time_period": self.sales_config.time_period,
                    "product_categories": self.sales_config.product_categories,
                    "regions": self.sales_config.regions,
                    "metrics": self.sales_config.metrics,
                    "segmentation": self.sales_config.segmentation
                },
                "data_fetching": self.data_fetching_results,
                "analysis": self.analysis_results,
                "recommendations": self.recommendation_results,
                "execution_summary": summary,
                "timestamp": datetime.datetime.now().isoformat()
            }

            return analysis_report

        except Exception as e:
            logger.error(f"Error processing sales analysis results: {str(e)}")
            return {
                "raw_result": result,
                "error": str(e),
                "timestamp": datetime.datetime.now().isoformat()
            }

    async def generate_executive_summary(self) -> str:
        """
        Generate an executive summary of the sales analysis.

        Returns:
            str: Executive summary
        """
        if not self.analysis_results or not self.recommendation_results:
            return "Analysis has not been completed yet."

        # Create a prompt for the strategic advisor to generate summary
        advisor = next((a for a in self.agents if isinstance(a, StrategicAdvisorAgent)), None)
        if not advisor:
            return "No strategic advisor agent available to generate summary."

        # Create a special task just for summary generation
        summary_task = Task(
            description=f"""
            Create a concise executive summary (max 250 words) of the sales analysis results and recommendations.

            The summary should:
            1. Highlight 3-5 key findings from the analysis
            2. Summarize the most important recommendations
            3. Focus on business impact and action items
            4. Be suitable for C-level executives

            Analysis context:
            - Time period: {self.sales_config.time_period}
            - Focus areas: {', '.join(self.sales_config.metrics)}
            - Business goals: {', '.join(self.business_context.get('strategic_goals', []))}
            """,
            agent=advisor.get_crew_agent(),
            expected_output="Concise executive summary"
        )

        # Execute the task with just this agent
        from crewai import Crew
        summary_crew = Crew(
            agents=[advisor.get_crew_agent()],
            tasks=[summary_task],
            verbose=self.config.verbose
        )

        summary = await summary_crew.run_async()
        return summary


# Factory function to create a sales analysis crew with default configuration
def create_sales_analysis_crew(**kwargs) -> SalesAnalysisCrew:
    """
    Create a Sales Analysis Crew with default configuration.

    Args:
        **kwargs: Override default configuration parameters

    Returns:
        SalesAnalysisCrew: Configured crew instance
    """
    return SalesAnalysisCrew(**kwargs)