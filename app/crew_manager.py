"""
Crew Manager for Snowflake Agent.
Handles crew initialization and execution.
"""

from datetime import datetime
from typing import Dict, Any, Optional

from crews.research_crew import BusinessResearchCrew, ResearchQuestion
from crews.analysis_crew import SalesAnalysisCrew, SalesDataConfig


def initialize_crew(
        crew_type: str,
        verbose: bool = False,
        time_period: Optional[str] = None,
        custom_start_date: Optional[datetime] = None,
        custom_end_date: Optional[datetime] = None
):
    """
    Initialize a crew based on the specified type and configuration.

    Args:
        crew_type: Type of crew to initialize ("Business Research" or "Sales Analysis")
        verbose: Whether to enable verbose output
        time_period: Time period for analysis (for Sales Analysis)
        custom_start_date: Custom start date (for Sales Analysis with custom time period)
        custom_end_date: Custom end date (for Sales Analysis with custom time period)

    Returns:
        The initialized crew instance
    """
    if crew_type == "Business Research":
        return BusinessResearchCrew(
            verbose=verbose
        )
    elif crew_type == "Sales Analysis":
        # Create sales config
        if time_period == "custom" and custom_start_date and custom_end_date:
            sales_config = SalesDataConfig(
                time_period="custom",
                custom_start_date=custom_start_date.strftime("%Y-%m-%d"),
                custom_end_date=custom_end_date.strftime("%Y-%m-%d")
            )
        elif time_period:
            sales_config = SalesDataConfig(
                time_period=time_period
            )
        else:
            sales_config = SalesDataConfig()

        return SalesAnalysisCrew(
            sales_config=sales_config,
            verbose=verbose
        )
    else:
        raise ValueError(f"Unknown crew type: {crew_type}")


async def run_crew_analysis(
        crew: Any,
        crew_type: str,
        question: str,
        context: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run analysis using the specified crew.

    Args:
        crew: The initialized crew instance
        crew_type: Type of crew ("Business Research" or "Sales Analysis")
        question: Business question to analyze
        context: Additional context for the question

    Returns:
        Dictionary containing analysis results
    """
    try:
        if crew_type == "Business Research":
            # Create research question
            research_q = ResearchQuestion(
                question=question,
                context=context if context else None,
                output_format="report"
            )

            # Run crew
            return await crew.answer_research_question(research_q)

        elif crew_type == "Sales Analysis":
            # For sales analysis, we don't need the context or to create a special object
            return await crew.execute_sales_analysis()

        else:
            raise ValueError(f"Unknown crew type: {crew_type}")

    except Exception as e:
        # Re-raise the exception with more context
        raise Exception(f"Error running crew analysis: {str(e)}") from e