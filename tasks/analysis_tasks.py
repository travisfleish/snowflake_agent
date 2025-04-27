"""
Business analytics tasks for the Snowflake Agent application.
Provides task classes for key business metrics analysis.
"""

import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta

from crewai import Task, Agent

from utils.snowflake_connector import connector
from utils.data_processors import SnowflakeDataProcessor

# Configure logger
logger = logging.getLogger(__name__)


class CustomerChurnAnalysisTask:
    """
    Analyzes customer churn patterns, rates, and risk factors using Snowflake data.
    """

    @staticmethod
    def create(
            time_period: str = "last_90_days",
            custom_start_date: str = None,
            custom_end_date: str = None,
            customer_segments: List[str] = None,
            include_features: bool = True,
            comparison_period: str = "previous_period",
            agent: Optional[Agent] = None,
            description: str = None
    ) -> Task:
        """
        Create a customer churn analysis task.

        Args:
            time_period: Predefined time period to analyze
                         ('last_30_days', 'last_90_days', 'last_year', 'custom')
            custom_start_date: Start date for custom period (YYYY-MM-DD)
            custom_end_date: End date for custom period (YYYY-MM-DD)
            customer_segments: List of customer segments to analyze separately
            include_features: Whether to include feature importance analysis
            comparison_period: Baseline period for comparison
                              ('previous_period', 'year_over_year', 'none')
            agent: Agent to assign the task to
            description: Custom task description

        Returns:
            Task: Configured task instance
        """
        # Calculate date ranges
        date_ranges = CustomerChurnAnalysisTask._calculate_date_ranges(
            time_period, custom_start_date, custom_end_date, comparison_period
        )

        # Generate task description
        if description is None:
            description = f"Analyze customer churn for {time_period}"
            if custom_start_date and custom_end_date:
                description = f"Analyze customer churn from {custom_start_date} to {custom_end_date}"

            if customer_segments:
                description += f" across segments: {', '.join(customer_segments)}"

            description += f"\n\nCompare with {comparison_period} to identify trends and patterns."

            if include_features:
                description += "\nIdentify key features and behaviors correlated with churn risk."

        # Define the SQL query to execute
        churn_query = CustomerChurnAnalysisTask._build_churn_query(
            date_ranges["current_start"],
            date_ranges["current_end"],
            customer_segments
        )

        # Define comparison query if needed
        comparison_query = None
        if comparison_period != "none":
            comparison_query = CustomerChurnAnalysisTask._build_churn_query(
                date_ranges["comparison_start"],
                date_ranges["comparison_end"],
                customer_segments
            )

        # Create the task
        return Task(
            description=description,
            agent=agent,
            expected_output="Comprehensive churn analysis with rates, patterns, and key risk factors",
            context={
                "task_type": "customer_churn_analysis",
                "time_period": time_period,
                "date_ranges": date_ranges,
                "customer_segments": customer_segments,
                "include_features": include_features,
                "comparison_period": comparison_period,
                "churn_query": churn_query,
                "comparison_query": comparison_query
            }
        )

    @staticmethod
    def _calculate_date_ranges(
            time_period: str,
            custom_start_date: str,
            custom_end_date: str,
            comparison_period: str
    ) -> Dict[str, str]:
        """Calculate appropriate date ranges based on parameters."""
        today = datetime.now()

        # Set current period
        if time_period == "custom" and custom_start_date and custom_end_date:
            current_start = custom_start_date
            current_end = custom_end_date
            current_days = (datetime.strptime(custom_end_date, "%Y-%m-%d") -
                            datetime.strptime(custom_start_date, "%Y-%m-%d")).days
        elif time_period == "last_30_days":
            current_end = today.strftime("%Y-%m-%d")
            current_start = (today - timedelta(days=30)).strftime("%Y-%m-%d")
            current_days = 30
        elif time_period == "last_90_days":
            current_end = today.strftime("%Y-%m-%d")
            current_start = (today - timedelta(days=90)).strftime("%Y-%m-%d")
            current_days = 90
        elif time_period == "last_year":
            current_end = today.strftime("%Y-%m-%d")
            current_start = (today - timedelta(days=365)).strftime("%Y-%m-%d")
            current_days = 365
        else:
            # Default to last 90 days
            current_end = today.strftime("%Y-%m-%d")
            current_start = (today - timedelta(days=90)).strftime("%Y-%m-%d")
            current_days = 90

        # Set comparison period
        if comparison_period == "previous_period":
            comparison_end = (datetime.strptime(current_start, "%Y-%m-%d") -
                              timedelta(days=1)).strftime("%Y-%m-%d")
            comparison_start = (datetime.strptime(comparison_end, "%Y-%m-%d") -
                                timedelta(days=current_days)).strftime("%Y-%m-%d")
        elif comparison_period == "year_over_year":
            comparison_start = (datetime.strptime(current_start, "%Y-%m-%d") -
                                timedelta(days=365)).strftime("%Y-%m-%d")
            comparison_end = (datetime.strptime(current_end, "%Y-%m-%d") -
                              timedelta(days=365)).strftime("%Y-%m-%d")
        else:
            comparison_start = None
            comparison_end = None

        return {
            "current_start": current_start,
            "current_end": current_end,
            "comparison_start": comparison_start,
            "comparison_end": comparison_end
        }

    @staticmethod
    def _build_churn_query(start_date: str, end_date: str, customer_segments: List[str] = None) -> str:
        """Build the SQL query for churn analysis."""
        # This is a simplified query that assumes certain table structure
        # In a real implementation, you would adapt this to your actual schema
        query = f"""
        WITH active_customers AS (
            SELECT
                customer_id,
                MAX(transaction_date) as last_transaction_date,
                COUNT(transaction_id) as transaction_count,
                SUM(amount) as total_spend,
                AVG(amount) as avg_transaction_value,
                customer_segment
            FROM
                transactions
            WHERE
                transaction_date BETWEEN '{start_date}' AND '{end_date}'
            GROUP BY
                customer_id, customer_segment
        ),

        customer_features AS (
            SELECT
                c.customer_id,
                c.customer_segment,
                c.signup_date,
                DATEDIFF('day', c.signup_date, CURRENT_DATE()) as customer_tenure_days,
                a.last_transaction_date,
                a.transaction_count,
                a.total_spend,
                a.avg_transaction_value,
                CASE 
                    WHEN DATEDIFF('day', a.last_transaction_date, '{end_date}') > 60 THEN 1
                    ELSE 0
                END as is_churned
            FROM
                customers c
            LEFT JOIN
                active_customers a ON c.customer_id = a.customer_id
        )

        SELECT
            customer_segment,
            COUNT(*) as total_customers,
            SUM(is_churned) as churned_customers,
            (SUM(is_churned) * 100.0 / COUNT(*)) as churn_rate,
            AVG(transaction_count) as avg_transactions,
            AVG(total_spend) as avg_spend,
            AVG(customer_tenure_days) as avg_tenure_days
        FROM
            customer_features
        """

        if customer_segments and len(customer_segments) > 0:
            segments_list = "', '".join(customer_segments)
            query += f"\nWHERE customer_segment IN ('{segments_list}')"

        query += "\nGROUP BY customer_segment ORDER BY churn_rate DESC"

        return query


class SalesByRegionTask:
    """
    Analyzes sales distribution, patterns, and trends by geographic region.
    """

    @staticmethod
    def create(
            time_period: str = "last_quarter",
            regions: List[str] = None,
            metrics: List[str] = None,
            compare_to_previous: bool = True,
            include_visualizations: bool = True,
            agent: Optional[Agent] = None,
            description: str = None
    ) -> Task:
        """
        Create a sales by region analysis task.

        Args:
            time_period: Time period to analyze
                         ('last_month', 'last_quarter', 'ytd', 'custom')
            regions: List of regions to include (None for all regions)
            metrics: List of metrics to calculate
            compare_to_previous: Whether to compare with previous period
            include_visualizations: Whether to include visualization specifications
            agent: Agent to assign the task to
            description: Custom task description

        Returns:
            Task: Configured task instance
        """
        # Set default metrics if none provided
        if metrics is None:
            metrics = ["revenue", "units_sold", "profit_margin", "average_transaction_value"]

        # Generate date range based on time period
        date_range = SalesByRegionTask._calculate_date_range(time_period)

        # Generate task description
        if description is None:
            description = f"Analyze sales by region for {time_period}"
            if regions:
                description += f" focusing on: {', '.join(regions)}"

            description += f"\n\nAnalyze the following metrics: {', '.join(metrics)}"

            if compare_to_previous:
                description += "\nCompare with previous period to identify trends."

            if include_visualizations:
                description += "\nInclude visualizations to represent regional performance."

        # Build the SQL query
        query = SalesByRegionTask._build_region_sales_query(
            date_range["start_date"],
            date_range["end_date"],
            regions,
            metrics
        )

        # Build comparison query if needed
        comparison_query = None
        if compare_to_previous:
            comparison_query = SalesByRegionTask._build_region_sales_query(
                date_range["comparison_start"],
                date_range["comparison_end"],
                regions,
                metrics
            )

        # Create the task
        return Task(
            description=description,
            agent=agent,
            expected_output="Regional sales analysis with performance metrics and trends",
            context={
                "task_type": "sales_by_region",
                "time_period": time_period,
                "date_range": date_range,
                "regions": regions,
                "metrics": metrics,
                "compare_to_previous": compare_to_previous,
                "include_visualizations": include_visualizations,
                "query": query,
                "comparison_query": comparison_query
            }
        )

    @staticmethod
    def _calculate_date_range(time_period: str) -> Dict[str, str]:
        """Calculate appropriate date range based on time period."""
        today = datetime.now()

        if time_period == "last_month":
            # Previous calendar month
            if today.month == 1:
                start_date = f"{today.year - 1}-12-01"
                end_date = f"{today.year - 1}-12-31"
            else:
                start_date = f"{today.year}-{today.month - 1:02d}-01"
                last_day = (datetime(today.year, today.month, 1) - timedelta(days=1)).day
                end_date = f"{today.year}-{today.month - 1:02d}-{last_day}"

            # Month before previous month for comparison
            if today.month <= 2:
                comparison_start = f"{today.year - 1}-{(12 if today.month == 1 else 11)}-01"
                last_day = (datetime(today.year, 1 if today.month == 1 else 12, 1) - timedelta(days=1)).day
                comparison_end = f"{today.year - 1}-{(12 if today.month == 1 else 11)}-{last_day}"
            else:
                comparison_start = f"{today.year}-{today.month - 2:02d}-01"
                last_day = (datetime(today.year, today.month - 1, 1) - timedelta(days=1)).day
                comparison_end = f"{today.year}-{today.month - 2:02d}-{last_day}"

        elif time_period == "last_quarter":
            # Previous quarter
            current_quarter = (today.month - 1) // 3 + 1
            previous_quarter = current_quarter - 1 if current_quarter > 1 else 4
            previous_year = today.year if current_quarter > 1 else today.year - 1

            start_month = (previous_quarter - 1) * 3 + 1
            start_date = f"{previous_year}-{start_month:02d}-01"

            end_month = previous_quarter * 3
            last_day = (datetime(previous_year, end_month + 1 if end_month < 12 else 1, 1) - timedelta(days=1)).day
            end_date = f"{previous_year}-{end_month:02d}-{last_day}"

            # Quarter before previous quarter for comparison
            second_previous_quarter = previous_quarter - 1 if previous_quarter > 1 else 4
            second_previous_year = previous_year if previous_quarter > 1 else previous_year - 1

            start_month = (second_previous_quarter - 1) * 3 + 1
            comparison_start = f"{second_previous_year}-{start_month:02d}-01"

            end_month = second_previous_quarter * 3
            last_day = (datetime(second_previous_year, end_month + 1 if end_month < 12 else 1, 1) - timedelta(
                days=1)).day
            comparison_end = f"{second_previous_year}-{end_month:02d}-{last_day}"

        elif time_period == "ytd":
            # Year to date (from Jan 1 to yesterday)
            start_date = f"{today.year}-01-01"
            yesterday = (today - timedelta(days=1)).strftime("%Y-%m-%d")
            end_date = yesterday

            # Same period last year for comparison
            comparison_start = f"{today.year - 1}-01-01"
            last_year_yesterday = datetime(today.year - 1,
                                           (today - timedelta(days=1)).month,
                                           (today - timedelta(days=1)).day).strftime("%Y-%m-%d")
            comparison_end = last_year_yesterday

        else:
            # Default to last 90 days
            end_date = (today - timedelta(days=1)).strftime("%Y-%m-%d")
            start_date = (today - timedelta(days=90)).strftime("%Y-%m-%d")

            # Previous 90 days for comparison
            comparison_end = (today - timedelta(days=91)).strftime("%Y-%m-%d")
            comparison_start = (today - timedelta(days=180)).strftime("%Y-%m-%d")

        return {
            "start_date": start_date,
            "end_date": end_date,
            "comparison_start": comparison_start,
            "comparison_end": comparison_end,
            "period_name": time_period
        }

    @staticmethod
    def _build_region_sales_query(
            start_date: str,
            end_date: str,
            regions: List[str] = None,
            metrics: List[str] = None
    ) -> str:
        """Build the SQL query for region sales analysis."""
        # Map metrics to SQL expressions
        metric_expressions = {
            "revenue": "SUM(amount)",
            "units_sold": "SUM(quantity)",
            "profit_margin": "SUM(profit) / SUM(amount) * 100",
            "average_transaction_value": "SUM(amount) / COUNT(DISTINCT transaction_id)",
            "order_count": "COUNT(DISTINCT transaction_id)",
            "customer_count": "COUNT(DISTINCT customer_id)"
        }

        # Select the requested metrics or default to revenue
        select_expressions = []
        for metric in (metrics or ["revenue"]):
            if metric in metric_expressions:
                select_expressions.append(f"{metric_expressions[metric]} as {metric}")

        # Build the query
        query = f"""
        SELECT
            region,
            {', '.join(select_expressions)}
        FROM
            sales_transactions
        WHERE
            transaction_date BETWEEN '{start_date}' AND '{end_date}'
        """

        # Add region filter if specified
        if regions and len(regions) > 0:
            regions_list = "', '".join(regions)
            query += f"\nAND region IN ('{regions_list}')"

        query += "\nGROUP BY region ORDER BY revenue DESC"

        return query


class TopPerformingProductsTask:
    """
    Identifies and analyzes top-performing products across various metrics.
    """

    @staticmethod
    def create(
            time_period: str = "last_90_days",
            product_categories: List[str] = None,
            ranking_metric: str = "revenue",
            top_n: int = 10,
            include_trend_analysis: bool = True,
            agent: Optional[Agent] = None,
            description: str = None
    ) -> Task:
        """
        Create a top performing products analysis task.

        Args:
            time_period: Time period to analyze
                         ('last_30_days', 'last_90_days', 'last_year', 'custom')
            product_categories: List of product categories to include (None for all)
            ranking_metric: Metric to rank products by
                           ('revenue', 'units_sold', 'profit_margin', 'growth')
            top_n: Number of top products to analyze
            include_trend_analysis: Whether to include trend analysis over time
            agent: Agent to assign the task to
            description: Custom task description

        Returns:
            Task: Configured task instance
        """
        # Generate date range based on time period
        date_range = TopPerformingProductsTask._calculate_date_range(time_period)

        # Generate task description
        if description is None:
            description = f"Identify top {top_n} performing products for {time_period}"
            if product_categories:
                description += f" in categories: {', '.join(product_categories)}"

            description += f"\n\nRank products by {ranking_metric}"

            if include_trend_analysis:
                description += "\nInclude trend analysis to show performance over time."

        # Build the SQL query
        query = TopPerformingProductsTask._build_top_products_query(
            date_range["start_date"],
            date_range["end_date"],
            product_categories,
            ranking_metric,
            top_n
        )

        # Build trend analysis query if needed
        trend_query = None
        if include_trend_analysis:
            trend_query = TopPerformingProductsTask._build_product_trend_query(
                date_range["start_date"],
                date_range["end_date"],
                product_categories,
                ranking_metric
            )

        # Create the task
        return Task(
            description=description,
            agent=agent,
            expected_output="Analysis of top performing products with key metrics and trends",
            context={
                "task_type": "top_performing_products",
                "time_period": time_period,
                "date_range": date_range,
                "product_categories": product_categories,
                "ranking_metric": ranking_metric,
                "top_n": top_n,
                "include_trend_analysis": include_trend_analysis,
                "query": query,
                "trend_query": trend_query
            }
        )

    @staticmethod
    def _calculate_date_range(time_period: str) -> Dict[str, str]:
        """Calculate appropriate date range based on time period."""
        today = datetime.now()

        if time_period == "last_30_days":
            end_date = (today - timedelta(days=1)).strftime("%Y-%m-%d")
            start_date = (today - timedelta(days=30)).strftime("%Y-%m-%d")
        elif time_period == "last_90_days":
            end_date = (today - timedelta(days=1)).strftime("%Y-%m-%d")
            start_date = (today - timedelta(days=90)).strftime("%Y-%m-%d")
        elif time_period == "last_year":
            end_date = (today - timedelta(days=1)).strftime("%Y-%m-%d")
            start_date = (today - timedelta(days=365)).strftime("%Y-%m-%d")
        else:
            # Default to last 90 days
            end_date = (today - timedelta(days=1)).strftime("%Y-%m-%d")
            start_date = (today - timedelta(days=90)).strftime("%Y-%m-%d")

        return {
            "start_date": start_date,
            "end_date": end_date,
            "period_name": time_period
        }

    @staticmethod
    def _build_top_products_query(
            start_date: str,
            end_date: str,
            product_categories: List[str] = None,
            ranking_metric: str = "revenue",
            top_n: int = 10
    ) -> str:
        """Build the SQL query for top products analysis."""
        # Map ranking metrics to SQL expressions
        metric_expressions = {
            "revenue": "SUM(amount)",
            "units_sold": "SUM(quantity)",
            "profit_margin": "SUM(profit) / SUM(amount) * 100",
            "growth": """
                (SUM(CASE WHEN transaction_date BETWEEN 
                    DATEADD('day', -DATEDIFF('day', '{start_date}', '{end_date}') / 2, '{end_date}') 
                    AND '{end_date}' THEN amount ELSE 0 END) - 
                 SUM(CASE WHEN transaction_date BETWEEN 
                    '{start_date}' 
                    AND DATEADD('day', -DATEDIFF('day', '{start_date}', '{end_date}') / 2 - 1, '{end_date}') 
                    THEN amount ELSE 0 END)) /
                NULLIF(SUM(CASE WHEN transaction_date BETWEEN 
                    '{start_date}' 
                    AND DATEADD('day', -DATEDIFF('day', '{start_date}', '{end_date}') / 2 - 1, '{end_date}') 
                    THEN amount ELSE 0 END), 0) * 100
            """
        }

        # Get ranking expression
        ranking_expr = metric_expressions.get(ranking_metric, metric_expressions["revenue"])
        if ranking_metric == "growth":
            ranking_expr = ranking_expr.replace("{start_date}", start_date).replace("{end_date}", end_date)

        # Build the query
        query = f"""
        SELECT
            p.product_id,
            p.product_name,
            p.product_category,
            {ranking_expr} as {ranking_metric},
            SUM(s.amount) as revenue,
            SUM(s.quantity) as units_sold,
            SUM(s.profit) / SUM(s.amount) * 100 as profit_margin,
            COUNT(DISTINCT s.transaction_id) as order_count
        FROM
            products p
        JOIN
            sales_transactions s ON p.product_id = s.product_id
        WHERE
            s.transaction_date BETWEEN '{start_date}' AND '{end_date}'
        """

        # Add category filter if specified
        if product_categories and len(product_categories) > 0:
            categories_list = "', '".join(product_categories)
            query += f"\nAND p.product_category IN ('{categories_list}')"

        query += f"\nGROUP BY p.product_id, p.product_name, p.product_category"
        query += f"\nORDER BY {ranking_metric} DESC"
        query += f"\nLIMIT {top_n}"

        return query

    @staticmethod
    def _build_product_trend_query(
            start_date: str,
            end_date: str,
            product_categories: List[str] = None,
            ranking_metric: str = "revenue"
    ) -> str:
        """Build the SQL query for product performance trends over time."""
        # Get the top products first
        top_products_query = TopPerformingProductsTask._build_top_products_query(
            start_date, end_date, product_categories, ranking_metric, 10
        )

        # Map metrics to SQL expressions
        metric_expressions = {
            "revenue": "SUM(s.amount)",
            "units_sold": "SUM(s.quantity)",
            "profit_margin": "SUM(s.profit) / SUM(s.amount) * 100",
            "order_count": "COUNT(DISTINCT s.transaction_id)"
        }

        # Get metric expression
        metric_expr = metric_expressions.get(ranking_metric, metric_expressions["revenue"])

        # Build the trend query
        trend_query = f"""
        WITH top_products AS (
            {top_products_query}
        )

        SELECT
            p.product_name,
            DATE_TRUNC('month', s.transaction_date) as month,
            {metric_expr} as {ranking_metric}
        FROM
            sales_transactions s
        JOIN
            products p ON s.product_id = p.product_id
        JOIN
            top_products tp ON p.product_id = tp.product_id
        WHERE
            s.transaction_date BETWEEN '{start_date}' AND '{end_date}'
        """

        # Add category filter if specified
        if product_categories and len(product_categories) > 0:
            categories_list = "', '".join(product_categories)
            trend_query += f"\nAND p.product_category IN ('{categories_list}')"

        trend_query += f"\nGROUP BY p.product_name, DATE_TRUNC('month', s.transaction_date)"
        trend_query += f"\nORDER BY p.product_name, month"

        return trend_query


# Register tasks with registry
from tasks.task_registry import register_task

# Register the CustomerChurnAnalysisTask
register_task(
    name="customer_churn_analysis",
    description="Analyzes customer churn patterns, rates, and risk factors",
    module_path="tasks.business_analytics_tasks",
    class_name="CustomerChurnAnalysisTask",
    parameters={
        "time_period": {
            "description": "Time period to analyze",
            "required": False,
            "type": "str",
            "default": "last_90_days"
        },
        "custom_start_date": {
            "description": "Start date for custom period (YYYY-MM-DD)",
            "required": False,
            "type": "str"
        },
        "custom_end_date": {
            "description": "End date for custom period (YYYY-MM-DD)",
            "required": False,
            "type": "str"
        },
        "customer_segments": {
            "description": "Customer segments to analyze separately",
            "required": False,
            "type": "List[str]"
        },
        "include_features": {
            "description": "Whether to include feature importance analysis",
            "required": False,
            "type": "bool",
            "default": True
        },
        "comparison_period": {
            "description": "Baseline period for comparison",
            "required": False,
            "type": "str",
            "default": "previous_period"
        }
    },
    category="business_analytics",
    tags=["customers", "churn", "retention"],
    factory_method="create"
)

# Register the SalesByRegionTask
register_task(
    name="sales_by_region",
    description="Analyzes sales distribution, patterns, and trends by geographic region",
    module_path="tasks.business_analytics_tasks",
    class_name="SalesByRegionTask",
    parameters={
        "time_period": {
            "description": "Time period to analyze",
            "required": False,
            "type": "str",
            "default": "last_quarter"
        },
        "regions": {
            "description": "List of regions to include",
            "required": False,
            "type": "List[str]"
        },
        "metrics": {
            "description": "List of metrics to calculate",
            "required": False,
            "type": "List[str]",
            "default": ["revenue", "units_sold", "profit_margin", "average_transaction_value"]
        },
        "compare_to_previous": {
            "description": "Whether to compare with previous period",
            "required": False,
            "type": "bool",
            "default": True
        },
        "include_visualizations": {
            "description": "Whether to include visualization specifications",
            "required": False,
            "type": "bool",
            "default": True
        }
    },
    category="business_analytics",
    tags=["sales", "regional", "geographic"],
    factory_method="create"
)

# Register the TopPerformingProductsTask
register_task(
    name="top_performing_products",
    description="Identifies and analyzes top-performing products across various metrics",
    module_path="tasks.business_analytics_tasks",
    class_name="TopPerformingProductsTask",
    parameters={
        "time_period": {
            "description": "Time period to analyze",
            "required": False,
            "type": "str",
            "default": "last_90_days"
        },
        "product_categories": {
            "description": "List of product categories to include",
            "required": False,
            "type": "List[str]"
        },
        "ranking_metric": {
            "description": "Metric to rank products by",
            "required": False,
            "type": "str",
            "default": "revenue"
        },
        "top_n": {
            "description": "Number of top products to analyze",
            "required": False,
            "type": "int",
            "default": 10
        },
        "include_trend_analysis": {
            "description": "Whether to include trend analysis over time",
            "required": False,
            "type": "bool",
            "default": True
        }
    },
    category="business_analytics",
    tags=["products", "performance", "ranking"],
    factory_method="create"
)