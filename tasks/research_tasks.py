from crews.base_crew import BaseCrew, CrewConfig
from agents.research_agent import SnowflakeAnalystAgent
from agents.analyst_agent import DataInsightAgent
from agents.executive_agent import StrategicAdvisorAgent
from crewai import Task
from typing import List, Dict, Any, Optional


class MarketTrendsCrew(BaseCrew):
    def __init__(
            self,
            name: str = "Market Trends Research Crew",
            description: str = "Identify market trends from historical sales data",
            time_period: str = "last_3_years",
            comparison_granularity: str = "quarterly",
            categories_to_analyze: List[str] = None,
            business_context: Dict[str, Any] = None,
            **kwargs
    ):
        # Create crew config
        config = CrewConfig(
            name=name,
            description=description,
            sequential=True,  # Tasks must be executed in order
            **kwargs
        )

        # Initialize base crew
        super().__init__(config=config)

        # Store parameters
        self.time_period = time_period
        self.comparison_granularity = comparison_granularity
        self.categories_to_analyze = categories_to_analyze or ["all"]

        # Set business context
        self.business_context = business_context or {
            "industry": "retail",
            "market_position": "mid-tier",
            "key_competitors": ["CompA", "CompB", "CompC"],
            "strategic_goals": ["market_expansion", "product_diversification"]
        }

        # Initialize default agents if none are provided
        if not self.agents:
            self._initialize_default_agents()

        # Set up the workflow tasks
        self.setup_market_trends_workflow()

    def _initialize_default_agents(self) -> None:
        """Initialize the specialized agents needed for market trend analysis."""
        # Data engineer for querying historical data
        data_engineer = SnowflakeAnalystAgent(
            name="Historical Data Analyst",
            role="SQL Expert & Data Engineer",
            goal="Extract comprehensive historical sales data with proper time series structure"
        )

        # Business analyst for trend identification
        trend_analyst = DataInsightAgent(
            name="Market Trend Analyst",
            role="Trend Identification Specialist",
            goal="Identify significant market trends, patterns, and shifts from historical data"
        )

        # Strategic advisor for competitive implications
        market_strategist = StrategicAdvisorAgent(
            name="Market Strategist",
            role="Market Intelligence Advisor",
            goal="Interpret market trends in competitive context and identify strategic opportunities"
        )

        # Add agents to the crew
        self.add_agents([data_engineer, trend_analyst, market_strategist])

    def setup_market_trends_workflow(self) -> None:
        """Set up the specialized tasks for market trend research."""
        # Get the agents
        data_engineer = self.agents[0]
        trend_analyst = self.agents[1]
        market_strategist = self.agents[2]

        # Task 1: Data Retrieval - Historical Sales Data
        data_retrieval_task = self.create_task(
            description=f"""
            Extract comprehensive historical sales data for market trend analysis with these specifications:

            Time period: {self.time_period}
            Comparison granularity: {self.comparison_granularity}
            Product categories: {', '.join(self.categories_to_analyze)}

            Your task requirements:

            1. Write optimized SQL queries to extract time-series sales data from the sales_transactions table
            2. Include critical dimensions: product_category, region, customer_segment, sales_channel
            3. Calculate key metrics over time: revenue, units_sold, average_order_value, customer_count
            4. Structure the data to enable proper time-series analysis with consistent time intervals
            5. Include market share data if available by joining with market_data tables
            6. Ensure data quality by validating time continuity and handling seasonality appropriately
            7. Document any data limitations, gaps, or quality issues that might affect trend analysis

            Output Format:
            - Provide the executed SQL queries with explanations
            - Include summary statistics of the retrieved dataset
            - Format the data for time-series analysis with proper date indexing
            """,
            agent=data_engineer,
            expected_output="Comprehensive historical sales dataset prepared for trend analysis"
        )

        # Task 2: Trend Identification and Analysis
        trend_analysis_task = self.create_task(
            description=f"""
            Analyze the historical sales data to identify significant market trends and patterns:

            Using the historical sales data provided by the Data Analyst, conduct a thorough trend analysis:

            1. Time-Series Analysis
               - Identify clear upward or downward trends across key metrics
               - Detect seasonal patterns and cyclical behaviors
               - Quantify growth rates and trajectory changes over the {self.time_period}

            2. Market Segment Analysis
               - Identify fastest growing and declining product categories
               - Analyze performance shifts across regions and territories
               - Evaluate channel performance trends (e.g., e-commerce vs. retail)
               - Assess customer segment behavior changes over time

            3. Statistical Pattern Recognition
               - Conduct decomposition analysis to separate trend from seasonality
               - Identify statistically significant trend shifts using appropriate tests
               - Detect anomalies or outliers that suggest market disruptions
               - Calculate correlation between different metrics to identify relationships

            4. Visualization & Discovery
               - Create clear visualizations showing the most important trend patterns
               - Use techniques like heatmaps to identify trend intersections
               - Generate forecasts using appropriate time-series methods

            Output Format:
            - Detailed analysis of the 5-7 most significant trends identified
            - Quantified metrics showing the magnitude of each trend
            - Visualizations demonstrating key patterns
            - Assessment of trend reliability and confidence level
            """,
            agent=trend_analyst,
            expected_output="Comprehensive market trend analysis with visualizations and statistical validation"
        )

        # Task 3: Strategic Interpretation and Recommendations
        strategic_interpretation_task = self.create_task(
            description=f"""
            Interpret the identified market trends in competitive context and develop strategic recommendations:

            Based on the comprehensive trend analysis, provide strategic context and recommendations:

            1. Competitive Positioning Analysis
               - Interpret how identified trends affect our market position
               - Compare our performance trends against known competitor performance
               - Identify areas where we are gaining or losing competitive advantage
               - Assess how industry-wide trends are affecting different market players

            2. Market Opportunity Identification
               - Highlight emerging market segments revealed by the trend analysis
               - Identify underserved customer needs suggested by trend patterns
               - Evaluate potential for new product/service offerings based on trends
               - Quantify the size and growth potential of opportunity areas

            3. Threat Assessment
               - Identify concerning trends that suggest market challenges
               - Evaluate disruptive forces indicated by unusual trend patterns
               - Assess competitive threats based on market share trend analysis
               - Quantify potential business impact of negative trends

            4. Strategic Recommendations
               - Develop 3-5 specific, actionable strategic recommendations
               - Prioritize opportunities based on alignment with {', '.join(self.business_context.get('strategic_goals', []))}
               - Suggest tactical responses to immediate market shifts
               - Outline long-term strategic adjustments based on sustained trends

            Output Format:
            - Executive summary of market position based on trend analysis
            - Strategic interpretation of each major trend identified
            - Prioritized list of opportunities and threats
            - Specific recommendations with rationale and expected outcomes
            - Implementation considerations for each recommendation
            """,
            agent=market_strategist,
            expected_output="Strategic market assessment with prioritized recommendations"
        )

        # Optional Task 4: Predictive Modeling (if you want to add this)
        # predictive_modeling_task = self.create_task(...)