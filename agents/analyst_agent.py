"""
DataInsightAgent for producing insights and summaries from Snowflake data.
Specializes in data analysis, visualization, and business intelligence.
"""

import logging
import json
import datetime
from typing import List, Dict, Any, Optional, Union
import pandas as pd
import numpy as np

from crewai import Task
from pydantic import BaseModel, Field

from agents.base_agent import BaseAgent
from tools.snowflake_tools import SnowflakeQueryTool, get_snowflake_tools
from utils.prompt_templates import SnowflakePromptTemplates
from utils.data_processors import SnowflakeDataProcessor

# Configure logger
logger = logging.getLogger(__name__)


class InsightConfiguration(BaseModel):
    """Configuration for insight generation."""

    insight_types: List[str] = Field(
        default=["summary", "trends", "anomalies", "recommendations"],
        description="Types of insights to generate"
    )
    detail_level: str = Field(
        default="medium",
        description="Level of detail for insights (low, medium, high)"
    )
    target_audience: str = Field(
        default="business",
        description="Target audience for insights (technical, business, executive)"
    )
    time_dimension: Optional[str] = Field(
        default=None,
        description="Time dimension column for trend analysis"
    )
    categoricals: List[str] = Field(
        default=[],
        description="Categorical columns for segmentation analysis"
    )
    metrics: List[str] = Field(
        default=[],
        description="Metric columns for numerical analysis"
    )
    visualization_types: List[str] = Field(
        default=["table", "line", "bar", "pie"],
        description="Types of visualizations to generate"
    )


class DataInsightAgent(BaseAgent):
    """
    Agent specialized in producing data insights and summaries from structured data.
    Analyzes Snowflake query results to identify patterns, outliers, and business implications.
    """

    def __init__(
            self,
            name: str = "Data Insight Analyst",
            role: str = "Data Analyst and Business Intelligence Expert",
            goal: str = "Generate valuable insights and visualizations from structured data",
            backstory: str = None,
            default_insight_config: Optional[InsightConfiguration] = None,
            **kwargs
    ):
        """
        Initialize a Data Insight Agent.

        Args:
            name: Agent's name
            role: Agent's role description
            goal: Agent's main objective
            backstory: Agent's background story (optional)
            default_insight_config: Default configuration for insight generation
            **kwargs: Additional agent parameters
        """
        # Generate detailed backstory if not provided
        if backstory is None:
            backstory = self._generate_insight_backstory()

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

        # Set default insight configuration
        self.default_insight_config = default_insight_config or InsightConfiguration()

        # Track data and insights cache
        self.data_cache = {}
        self.insight_cache = {}

        logger.info(f"Initialized {self.__class__.__name__}: {self.name}")

    def _generate_insight_backstory(self) -> str:
        """
        Generate a detailed backstory for a data insight analyst.

        Returns:
            str: Detailed backstory
        """
        return (
            "I am an experienced data analyst with expertise in transforming raw data into actionable business insights. "
            "With a background spanning business intelligence, data science, and strategic consulting, "
            "I excel at identifying meaningful patterns, anomalies, and trends in complex datasets. "
            "I have a talent for translating technical findings into clear, compelling narratives that drive decision-making. "
            "My expertise includes statistical analysis, data visualization, and communicating insights to diverse audiences "
            "from technical teams to executive leadership. I pride myself on my ability to look beyond the numbers "
            "to uncover the business implications and strategic opportunities hidden in the data."
        )

    async def analyze_data(
            self,
            data: Union[str, pd.DataFrame, Dict[str, Any]],
            context: str,
            config: Optional[InsightConfiguration] = None
    ) -> Dict[str, Any]:
        """
        Analyze data and generate insights.

        Args:
            data: Data to analyze (DataFrame, dict, or Snowflake results string)
            context: Business context for the analysis
            config: Custom insight configuration (uses default if None)

        Returns:
            Dict[str, Any]: Generated insights and visualizations
        """
        # Use provided config or default
        config = config or self.default_insight_config

        # Convert data to DataFrame if necessary
        df = self._ensure_dataframe(data)

        # Cache the data with timestamp
        data_id = f"data_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
        self.data_cache[data_id] = {
            "data": df,
            "context": context,
            "timestamp": datetime.datetime.now().isoformat()
        }

        # Generate data profile
        profile = self._generate_data_profile(df)

        # Generate insights based on configuration
        insights = {}

        if "summary" in config.insight_types:
            insights["summary"] = await self._generate_summary(df, context, config)

        if "trends" in config.insight_types and config.time_dimension:
            insights["trends"] = await self._analyze_trends(df, context, config)

        if "anomalies" in config.insight_types:
            insights["anomalies"] = await self._detect_anomalies(df, context, config)

        if "segmentation" in config.insight_types and config.categoricals:
            insights["segmentation"] = await self._perform_segmentation(df, context, config)

        if "correlations" in config.insight_types and len(config.metrics) > 1:
            insights["correlations"] = await self._analyze_correlations(df, context, config)

        if "recommendations" in config.insight_types:
            insights["recommendations"] = await self._generate_recommendations(
                df, insights, context, config
            )

        # Generate visualizations
        visualizations = await self._generate_visualizations(df, insights, config)

        # Create complete insight package
        result = {
            "data_id": data_id,
            "timestamp": datetime.datetime.now().isoformat(),
            "context": context,
            "profile": profile,
            "insights": insights,
            "visualizations": visualizations,
            "narrative": await self._create_narrative(insights, context, config)
        }

        # Cache insights
        self.insight_cache[data_id] = result

        return result

    def _ensure_dataframe(self, data: Union[str, pd.DataFrame, Dict[str, Any]]) -> pd.DataFrame:
        """
        Convert input data to pandas DataFrame.

        Args:
            data: Input data in various formats

        Returns:
            pd.DataFrame: Data as DataFrame
        """
        if isinstance(data, pd.DataFrame):
            return data

        if isinstance(data, dict):
            return pd.DataFrame(data)

        if isinstance(data, str):
            try:
                # Try to parse as Snowflake result string
                # This is a simplified approach - actual implementation would be more robust
                lines = data.strip().split('\n')

                # Try to determine if it's a formatted table
                if '|' in data or len(lines) > 1 and all(len(line.split()) > 1 for line in lines[:2]):
                    # Handle formatted table output
                    import io
                    if '|' in data:
                        # Markdown-like table
                        # Extract header and data rows
                        rows = []
                        header = None
                        for line in lines:
                            if line.strip() and not line.startswith('+-') and not line.startswith('--'):
                                cells = [cell.strip() for cell in line.split('|')[1:-1]]
                                if header is None:
                                    header = cells
                                else:
                                    rows.append(cells)

                        return pd.DataFrame(rows, columns=header)
                    else:
                        # Space-separated table
                        headers = lines[0].split()
                        data_rows = [line.split() for line in lines[1:] if line.strip()]
                        return pd.DataFrame(data_rows, columns=headers)

                # Try to parse as JSON
                import json
                try:
                    json_data = json.loads(data)
                    return pd.DataFrame(json_data)
                except:
                    pass

                # Try to parse as CSV
                try:
                    import io
                    return pd.read_csv(io.StringIO(data))
                except:
                    pass

                # If all else fails, return empty DataFrame with warning
                logger.warning("Could not parse data string, returning empty DataFrame")
                return pd.DataFrame()

            except Exception as e:
                logger.error(f"Error converting data to DataFrame: {str(e)}")
                return pd.DataFrame()

        # Fallback for unknown types
        logger.warning(f"Unknown data type {type(data)}, returning empty DataFrame")
        return pd.DataFrame()

    def _generate_data_profile(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate a profile of the dataset.

        Args:
            df: DataFrame to profile

        Returns:
            Dict[str, Any]: Dataset profile
        """
        try:
            # Get basic info
            profile = {
                "shape": df.shape,
                "columns": df.columns.tolist(),
                "dtypes": {col: str(df[col].dtype) for col in df.columns},
                "missing_values": {col: int(df[col].isna().sum()) for col in df.columns},
                "missing_percentage": {col: float(df[col].isna().mean() * 100) for col in df.columns},
            }

            # Add numeric summary
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            if numeric_cols:
                profile["numeric_summary"] = {}
                for col in numeric_cols:
                    profile["numeric_summary"][col] = {
                        "min": float(df[col].min()) if not df[col].isna().all() else None,
                        "max": float(df[col].max()) if not df[col].isna().all() else None,
                        "mean": float(df[col].mean()) if not df[col].isna().all() else None,
                        "median": float(df[col].median()) if not df[col].isna().all() else None,
                        "std": float(df[col].std()) if not df[col].isna().all() else None
                    }

            # Add categorical summary
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            if categorical_cols:
                profile["categorical_summary"] = {}
                for col in categorical_cols:
                    value_counts = df[col].value_counts()
                    profile["categorical_summary"][col] = {
                        "unique_count": int(df[col].nunique()),
                        "top_values": {
                            str(k): int(v) for k, v in value_counts.head(5).items()
                        }
                    }

            # Add datetime summary
            date_cols = df.select_dtypes(include=['datetime']).columns.tolist()
            if date_cols:
                profile["date_summary"] = {}
                for col in date_cols:
                    profile["date_summary"][col] = {
                        "min": df[col].min().isoformat() if not df[col].isna().all() else None,
                        "max": df[col].max().isoformat() if not df[col].isna().all() else None,
                        "range_days": int((df[col].max() - df[col].min()).days) if not df[col].isna().all() else None
                    }

            return profile

        except Exception as e:
            logger.error(f"Error generating data profile: {str(e)}")
            return {"error": str(e)}

    async def _generate_summary(
            self,
            df: pd.DataFrame,
            context: str,
            config: InsightConfiguration
    ) -> str:
        """
        Generate a summary of the dataset.

        Args:
            df: DataFrame to summarize
            context: Business context
            config: Insight configuration

        Returns:
            str: Data summary
        """
        # Create prompt for LLM
        profile = self._generate_data_profile(df)

        # Sample data (first few rows)
        sample_data = df.head(5).to_string()

        prompt = f"""
        You are a data analyst tasked with summarizing a dataset in the context of this business question: 
        "{context}"

        Dataset information:
        - Rows: {profile['shape'][0]}
        - Columns: {profile['shape'][1]}
        - Column names: {', '.join(profile['columns'])}

        Here's a sample of the data:
        {sample_data}

        Please provide a concise summary of this dataset that addresses:
        1. What kind of data is this? What entities does it represent?
        2. What is the scope of the data (time period, categories, metrics)?
        3. What are the key metrics and dimensions?
        4. What is the overall data quality (completeness, potential issues)?
        5. How well does this data address the business context?

        Keep your response focused, clear, and appropriate for a {config.target_audience} audience.
        Use a {config.detail_level} level of detail.
        """

        # Call LLM to generate summary
        self.thinking(f"Generating data summary for context: {context}")
        response = await self._llm_client.generate(
            prompt=prompt,
            temperature=0.4,
            max_tokens=500
        )

        return response['choices'][0]['message']['content']

    async def _analyze_trends(
            self,
            df: pd.DataFrame,
            context: str,
            config: InsightConfiguration
    ) -> Dict[str, Any]:
        """
        Analyze trends in time series data.

        Args:
            df: DataFrame with time dimension
            context: Business context
            config: Insight configuration

        Returns:
            Dict[str, Any]: Trend analysis
        """
        # Check if time dimension is present
        if not config.time_dimension or config.time_dimension not in df.columns:
            return {"error": "No valid time dimension specified"}

        try:
            # Ensure time dimension is datetime
            if not pd.api.types.is_datetime64_dtype(df[config.time_dimension]):
                try:
                    df[config.time_dimension] = pd.to_datetime(df[config.time_dimension])
                except:
                    return {"error": "Could not convert time dimension to datetime"}

            # Get metrics for trend analysis
            metrics = config.metrics or df.select_dtypes(include=['number']).columns.tolist()

            # Prepare trend data
            trend_data = {}
            for metric in metrics:
                if metric in df.columns:
                    # Group by time dimension and calculate mean
                    try:
                        # Determine appropriate time frequency based on date range
                        date_range = (df[config.time_dimension].max() - df[config.time_dimension].min()).days
                        if date_range > 365:
                            freq = 'M'  # Monthly
                        elif date_range > 30:
                            freq = 'W'  # Weekly
                        else:
                            freq = 'D'  # Daily

                        # Group and resample
                        metric_series = df.groupby(pd.Grouper(key=config.time_dimension, freq=freq))[metric].mean()

                        trend_data[metric] = {
                            "time_points": [d.isoformat() for d in metric_series.index],
                            "values": metric_series.tolist(),
                            "growth": float(((metric_series.iloc[-1] / metric_series.iloc[0]) - 1) * 100)
                            if len(metric_series) > 1 and metric_series.iloc[0] != 0 else None
                        }
                    except Exception as e:
                        logger.error(f"Error analyzing trend for {metric}: {str(e)}")

            # Generate trend narrative with LLM
            trend_narrative = await self._generate_trend_narrative(trend_data, context, config)

            return {
                "trends": trend_data,
                "narrative": trend_narrative
            }

        except Exception as e:
            logger.error(f"Error in trend analysis: {str(e)}")
            return {"error": str(e)}

    async def _generate_trend_narrative(
            self,
            trend_data: Dict[str, Any],
            context: str,
            config: InsightConfiguration
    ) -> str:
        """
        Generate narrative description of trends.

        Args:
            trend_data: Trend analysis data
            context: Business context
            config: Insight configuration

        Returns:
            str: Trend narrative
        """
        # Prepare trend summary for LLM
        trend_summary = []
        for metric, data in trend_data.items():
            if "growth" in data and data["growth"] is not None:
                trend_summary.append(f"{metric}: {data['growth']:.2f}% growth over the period")

        trend_summary_text = "\n".join(trend_summary)

        prompt = f"""
        You are analyzing trends in time series data for this business context:
        "{context}"

        The following metrics show these growth patterns:
        {trend_summary_text}

        Please provide a clear interpretation of these trends addressing:
        1. What are the most significant trends?
        2. Are there any seasonal patterns or cyclical behaviors?
        3. What might these trends mean for the business?
        4. What should decision-makers pay attention to?

        Keep your response focused, clear, and appropriate for a {config.target_audience} audience.
        Use a {config.detail_level} level of detail.
        """

        # Call LLM to generate trend narrative
        response = await self._llm_client.generate(
            prompt=prompt,
            temperature=0.4,
            max_tokens=500
        )

        return response['choices'][0]['message']['content']

    async def _detect_anomalies(
            self,
            df: pd.DataFrame,
            context: str,
            config: InsightConfiguration
    ) -> Dict[str, Any]:
        """
        Detect anomalies in the dataset.

        Args:
            df: DataFrame to analyze
            context: Business context
            config: Insight configuration

        Returns:
            Dict[str, Any]: Anomaly detection results
        """
        # Get metrics for anomaly detection
        metrics = config.metrics or df.select_dtypes(include=['number']).columns.tolist()

        anomalies = {}

        # Simple Z-score based anomaly detection
        for metric in metrics:
            if metric in df.columns:
                try:
                    # Calculate z-scores
                    mean = df[metric].mean()
                    std = df[metric].std()

                    if std > 0:  # Avoid division by zero
                        z_scores = (df[metric] - mean) / std

                        # Find outliers (z-score > 3 or < -3)
                        threshold = 3.0
                        outliers = df[abs(z_scores) > threshold]

                        if len(outliers) > 0:
                            # Record anomalies
                            anomalies[metric] = {
                                "count": len(outliers),
                                "percentage": float(len(outliers) / len(df) * 100),
                                "examples": outliers.head(3).to_dict('records'),
                                "threshold": f"Z-score > {threshold}"
                            }
                except Exception as e:
                    logger.error(f"Error detecting anomalies for {metric}: {str(e)}")

        # Generate anomaly narrative with LLM
        anomaly_narrative = await self._generate_anomaly_narrative(anomalies, context, config)

        return {
            "anomalies": anomalies,
            "narrative": anomaly_narrative
        }

    async def _generate_anomaly_narrative(
            self,
            anomalies: Dict[str, Any],
            context: str,
            config: InsightConfiguration
    ) -> str:
        """
        Generate narrative description of anomalies.

        Args:
            anomalies: Anomaly detection results
            context: Business context
            config: Insight configuration

        Returns:
            str: Anomaly narrative
        """
        # Prepare anomaly summary for LLM
        if not anomalies:
            return "No significant anomalies were detected in the data."

        anomaly_summary = []
        for metric, data in anomalies.items():
            anomaly_summary.append(f"{metric}: {data['count']} anomalies ({data['percentage']:.2f}% of data)")

        anomaly_summary_text = "\n".join(anomaly_summary)

        prompt = f"""
        You are analyzing anomalies in data for this business context:
        "{context}"

        The following metrics show these anomaly patterns:
        {anomaly_summary_text}

        Please provide a clear interpretation of these anomalies addressing:
        1. What do these anomalies represent in business terms?
        2. How significant are these anomalies?
        3. What might be causing these anomalies?
        4. What actions should be considered regarding these anomalies?

        Keep your response focused, clear, and appropriate for a {config.target_audience} audience.
        Use a {config.detail_level} level of detail.
        """

        # Call LLM to generate anomaly narrative
        response = await self._llm_client.generate(
            prompt=prompt,
            temperature=0.4,
            max_tokens=500
        )

        return response['choices'][0]['message']['content']

    async def _perform_segmentation(
            self,
            df: pd.DataFrame,
            context: str,
            config: InsightConfiguration
    ) -> Dict[str, Any]:
        """
        Perform segmentation analysis.

        Args:
            df: DataFrame to analyze
            context: Business context
            config: Insight configuration

        Returns:
            Dict[str, Any]: Segmentation results
        """
        # Get categorical columns for segmentation
        categoricals = config.categoricals or df.select_dtypes(include=['object', 'category']).columns.tolist()

        # Get metrics for analysis
        metrics = config.metrics or df.select_dtypes(include=['number']).columns.tolist()

        segmentation = {}

        # Analyze each categorical dimension
        for categorical in categoricals:
            if categorical in df.columns:
                try:
                    # Get top segments by frequency
                    value_counts = df[categorical].value_counts().head(5)
                    top_segments = value_counts.index.tolist()

                    segment_data = {
                        "count": value_counts.to_dict(),
                        "percentage": (value_counts / len(df) * 100).to_dict(),
                        "metrics": {}
                    }

                    # Calculate metrics by segment
                    for metric in metrics:
                        if metric in df.columns:
                            segment_metrics = {}

                            for segment in top_segments:
                                segment_df = df[df[categorical] == segment]
                                segment_metrics[str(segment)] = {
                                    "mean": float(segment_df[metric].mean()),
                                    "median": float(segment_df[metric].median()),
                                    "min": float(segment_df[metric].min()),
                                    "max": float(segment_df[metric].max())
                                }

                            segment_data["metrics"][metric] = segment_metrics

                    segmentation[categorical] = segment_data

                except Exception as e:
                    logger.error(f"Error in segmentation for {categorical}: {str(e)}")

        # Generate segmentation narrative with LLM
        segmentation_narrative = await self._generate_segmentation_narrative(segmentation, context, config)

        return {
            "segmentation": segmentation,
            "narrative": segmentation_narrative
        }

    async def _generate_segmentation_narrative(
            self,
            segmentation: Dict[str, Any],
            context: str,
            config: InsightConfiguration
    ) -> str:
        """
        Generate narrative description of segmentation.

        Args:
            segmentation: Segmentation results
            context: Business context
            config: Insight configuration

        Returns:
            str: Segmentation narrative
        """
        if not segmentation:
            return "No meaningful segmentation could be performed on the data."

        # Prepare segmentation summary for LLM
        segments_summary = []

        for dimension, data in segmentation.items():
            top_segment = max(data["count"].items(), key=lambda x: x[1])
            segments_summary.append(
                f"{dimension}: Top segment is '{top_segment[0]}' with {top_segment[1]} records ({data['percentage'][top_segment[0]]:.2f}%)")

            # Add metric differences if available
            if "metrics" in data and data["metrics"]:
                for metric_name, metric_data in data["metrics"].items():
                    # Find segment with highest mean
                    if metric_data:
                        top_metric_segment = max(metric_data.items(), key=lambda x: x[1]["mean"])
                        segments_summary.append(
                            f"  - For {metric_name}, '{top_metric_segment[0]}' has highest mean: {top_metric_segment[1]['mean']:.2f}")

        segments_summary_text = "\n".join(segments_summary)

        prompt = f"""
        You are analyzing segment differences in data for this business context:
        "{context}"

        The following segments were identified:
        {segments_summary_text}

        Please provide a clear interpretation of these segments addressing:
        1. What are the most significant segment differences?
        2. What do these differences mean in business terms?
        3. Which segments should receive special attention and why?
        4. What actions might be considered based on these segment insights?

        Keep your response focused, clear, and appropriate for a {config.target_audience} audience.
        Use a {config.detail_level} level of detail.
        """

        # Call LLM to generate segmentation narrative
        response = await self._llm_client.generate(
            prompt=prompt,
            temperature=0.4,
            max_tokens=500
        )

        return response['choices'][0]['message']['content']

    async def _analyze_correlations(
            self,
            df: pd.DataFrame,
            context: str,
            config: InsightConfiguration
    ) -> Dict[str, Any]:
        """
        Analyze correlations between metrics.

        Args:
            df: DataFrame to analyze
            context: Business context
            config: Insight configuration

        Returns:
            Dict[str, Any]: Correlation analysis
        """
        # Get metrics for correlation analysis
        metrics = config.metrics or df.select_dtypes(include=['number']).columns.tolist()

        # Need at least 2 metrics for correlation
        if len(metrics) < 2:
            return {"error": "Need at least 2 numeric metrics for correlation analysis"}

        correlations = {}

        try:
            # Calculate correlation matrix
            corr_df = df[metrics].corr()

            # Convert to dictionary format
            corr_dict = {}
            for metric1 in metrics:
                corr_dict[metric1] = {}
                for metric2 in metrics:
                    corr_dict[metric1][metric2] = float(corr_df.loc[metric1, metric2])

            # Find strong correlations (absolute value > 0.5)
            strong_correlations = []
            for i, metric1 in enumerate(metrics):
                for j, metric2 in enumerate(metrics[i + 1:], i + 1):
                    corr_value = corr_df.loc[metric1, metric2]
                    if abs(corr_value) > 0.5:
                        strong_correlations.append({
                            "metric1": metric1,
                            "metric2": metric2,
                            "correlation": float(corr_value),
                            "strength": "strong positive" if corr_value > 0.7 else
                            ("moderate positive" if corr_value > 0.5 else
                             ("strong negative" if corr_value < -0.7 else "moderate negative"))
                        })

            correlations = {
                "matrix": corr_dict,
                "strong_correlations": strong_correlations
            }

            # Generate correlation narrative with LLM
            correlation_narrative = await self._generate_correlation_narrative(correlations, context, config)

            correlations["narrative"] = correlation_narrative

            return correlations

        except Exception as e:
            logger.error(f"Error in correlation analysis: {str(e)}")
            return {"error": str(e)}

    async def _generate_correlation_narrative(
            self,
            correlations: Dict[str, Any],
            context: str,
            config: InsightConfiguration
    ) -> str:
        """
        Generate narrative description of correlations.

        Args:
            correlations: Correlation analysis
            context: Business context
            config: Insight configuration

        Returns:
            str: Correlation narrative
        """
        if "error" in correlations:
            return f"Could not perform correlation analysis: {correlations['error']}"

        if not correlations.get("strong_correlations"):
            return "No strong correlations were found between the metrics in this dataset."

        # Prepare correlation summary for LLM
        corr_summary = []

        for corr in correlations["strong_correlations"]:
            corr_summary.append(
                f"{corr['metric1']} and {corr['metric2']}: {corr['correlation']:.2f} ({corr['strength']})"
            )

        corr_summary_text = "\n".join(corr_summary)

        prompt = f"""
        You are analyzing correlations between metrics for this business context:
        "{context}"

        The following strong correlations were detected:
        {corr_summary_text}

        Please provide a clear interpretation of these correlations addressing:
        1. What do these correlations mean in business terms?
        2. Which correlations are most significant for the business?
        3. Do any correlations suggest causal relationships worth investigating?
        4. What actions or decisions might be informed by these correlations?

        Keep your response focused, clear, and appropriate for a {config.target_audience} audience.
        Use a {config.detail_level} level of detail.
        """

        # Call LLM to generate correlation narrative
        response = await self._llm_client.generate(
            prompt=prompt,
            temperature=0.4,
            max_tokens=500
        )

        return response['choices'][0]['message']['content']

    async def _generate_recommendations(
            self,
            df: pd.DataFrame,
            insights: Dict[str, Any],
            context: str,
            config: InsightConfiguration
    ) -> List[Dict[str, str]]:
        """
        Generate business recommendations based on insights.

        Args:
            df: DataFrame analyzed
            insights: Insights generated
            context: Business context
            config: Insight configuration

        Returns:
            List[Dict[str, str]]: Business recommendations
        """
        # Compile insights for LLM
        insight_summary = []

        if "summary" in insights:
            insight_summary.append(f"Data Summary: {insights['summary']}")

        if "trends" in insights and "narrative" in insights["trends"]:
            insight_summary.append(f"Trends: {insights['trends']['narrative']}")

        if "anomalies" in insights and "narrative" in insights["anomalies"]:
            insight_summary.append(f"Anomalies: {insights['anomalies']['narrative']}")

        if "segmentation" in insights and "narrative" in insights["segmentation"]:
            insight_summary.append(f"Segmentation: {insights['segmentation']['narrative']}")

        if "correlations" in insights and "narrative" in insights["correlations"]:
            insight_summary.append(f"Correlations: {insights['correlations']['narrative']}")

        insight_summary_text = "\n\n".join(insight_summary)

        prompt = f"""
        You are a business advisor generating recommendations based on data insights.
        The business context is: "{context}"

        Based on the following insights:

        {insight_summary_text}

        Generate 3-5 specific, actionable business recommendations. For each recommendation:
        1. Provide a clear, concise title
        2. Explain the recommendation in 2-3 sentences
        3. Explain the expected business impact
        4. Suggest how to measure the success of implementing this recommendation

        Format each recommendation as a JSON object with fields:
        - title: Clear title of the recommendation
        - description: 2-3 sentence explanation
        - impact: Expected business impact
        - measurement: How to measure success

        Return a JSON array of these recommendation objects.
        """

        # Call LLM to generate recommendations
        response = await self._llm_client.generate(
            prompt=prompt,
            temperature=0.5,
            max_tokens=1000
        )

        # Parse recommendations from response
        try:
            response_text = response['choices'][0]['message']['content']

            # Extract JSON array
            import re
            import json

            # Try to find JSON array in response
            json_match = re.search(r'\[\s*{.*}\s*\]', response_text, re.DOTALL)
            if json_match:
                recommendations = json.loads(json_match.group(0))
            else:
                # Fallback: Try to parse the whole response
                recommendations = json.loads(response_text)

            return recommendations

        except Exception as e:
            logger.error(f"Error parsing recommendations: {str(e)}")

            # Fallback: Create recommendations from raw text
            fallback_recommendation = {
                "title": "Recommendation",
                "description": response['choices'][0]['message']['content'],
                "impact": "See description",
                "measurement": "See description"
            }

            return [fallback_recommendation]

    async def _generate_visualizations(
            self,
            df: pd.DataFrame,
            insights: Dict[str, Any],
            config: InsightConfiguration
    ) -> Dict[str, str]:
        """
        Generate visualization specifications.

        Args:
            df: DataFrame analyzed
            insights: Insights generated
            config: Insight configuration

        Returns:
            Dict[str, str]: Visualization specifications
        """
        visualizations = {}

        # Generate visualization specs for different insight types
        try:
            # 1. Summary visualization (data distribution)
            if "table" in config.visualization_types:
                # Create summary table spec
                metrics = config.metrics or df.select_dtypes(include=['number']).columns.tolist()
                if metrics:
                    visualizations["summary_table"] = {
                        "type": "table",
                        "title": "Data Summary",
                        "data": {
                            "columns": metrics,
                            "metrics": ["mean", "median", "min", "max", "std"]
                        }
                    }

            # 2. Time series visualization
            if ("line" in config.visualization_types and
                    config.time_dimension and
                    "trends" in insights and
                    "trends" in insights["trends"]):

                trend_data = insights["trends"]["trends"]
                if trend_data:
                    # Get metric with highest growth
                    top_metric = max(trend_data.items(),
                                     key=lambda x: x[1].get("growth", 0)
                                     if x[1].get("growth") is not None else -float('inf'))

                    visualizations["trend_chart"] = {
                        "type": "line",
                        "title": f"Trend Analysis: {top_metric[0]}",
                        "data": {
                            "x": top_metric[1]["time_points"],
                            "y": top_metric[1]["values"],
                            "labels": {
                                "x": config.time_dimension,
                                "y": top_metric[0]
                            }
                        }
                    }

            # 3. Segmentation visualization
            if ("bar" in config.visualization_types and
                    "segmentation" in insights and
                    "segmentation" in insights["segmentation"]):

                seg_data = insights["segmentation"]["segmentation"]
                if seg_data:
                    # Get first segmentation dimension
                    seg_dim = next(iter(seg_data))

                    # Get count data
                    counts = seg_data[seg_dim]["count"]

                    visualizations["segment_chart"] = {
                        "type": "bar",
                        "title": f"Segment Distribution: {seg_dim}",
                        "data": {
                            "x": list(counts.keys()),
                            "y": list(counts.values()),
                            "labels": {
                                "x": seg_dim,
                                "y": "Count"
                            }
                        }
                    }

            # 4. Correlation visualization
            if ("heatmap" in config.visualization_types and
                    "correlations" in insights and
                    "matrix" in insights["correlations"]):

                corr_matrix = insights["correlations"]["matrix"]
                if corr_matrix:
                    visualizations["correlation_chart"] = {
                        "type": "heatmap",
                        "title": "Correlation Matrix",
                        "data": {
                            "matrix": corr_matrix
                        }
                    }

            return visualizations

        except Exception as e:
            logger.error(f"Error generating visualizations: {str(e)}")
            return {}

    async def _create_narrative(
            self,
            insights: Dict[str, Any],
            context: str,
            config: InsightConfiguration
    ) -> str:
        """
        Create overall narrative summary of all insights.

        Args:
            insights: All generated insights
            context: Business context
            config: Insight configuration

        Returns:
            str: Overall narrative
        """
        # Compile key points from all insights
        narrative_points = []

        if "summary" in insights:
            narrative_points.append(insights["summary"])

        if "trends" in insights and "narrative" in insights["trends"]:
            narrative_points.append(insights["trends"]["narrative"])

        if "anomalies" in insights and "narrative" in insights["anomalies"]:
            narrative_points.append(insights["anomalies"]["narrative"])

        if "segmentation" in insights and "narrative" in insights["segmentation"]:
            narrative_points.append(insights["segmentation"]["narrative"])

        if "correlations" in insights and "narrative" in insights["correlations"]:
            narrative_points.append(insights["correlations"]["narrative"])

        # Get recommendations if available
        recommendations = insights.get("recommendations", [])
        if recommendations:
            rec_text = "Key recommendations:\n"
            for i, rec in enumerate(recommendations, 1):
                rec_text += f"{i}. {rec.get('title')}: {rec.get('description')}\n"
            narrative_points.append(rec_text)

        # Create prompt for LLM
        narrative_text = "\n\n".join(narrative_points)

        prompt = f"""
        You are a data insights specialist creating an executive summary for this business context:
        "{context}"

        Based on these detailed insights:

        {narrative_text}

        Create a clear, concise executive summary that:
        1. Addresses the core business context
        2. Highlights the 3-5 most important findings
        3. Summarizes key recommendations
        4. Suggests next steps

        The summary should be appropriate for a {config.target_audience} audience using a {config.detail_level} level of detail.
        Keep it concise, impactful, and actionable.
        """

        # Call LLM to generate overall narrative
        response = await self._llm_client.generate(
            prompt=prompt,
            temperature=0.4,
            max_tokens=800
        )

        return response['choices'][0]['message']['content']

    async def get_insight_by_id(self, insight_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve previously generated insight by ID.

        Args:
            insight_id: ID of the insight to retrieve

        Returns:
            Optional[Dict[str, Any]]: Retrieved insight or None if not found
        """
        return self.insight_cache.get(insight_id)

    async def generate_report_for_insight(
            self,
            insight_id: str,
            report_type: str = "executive",
            custom_sections: List[str] = None
    ) -> str:
        """
        Generate a formatted report from an insight.

        Args:
            insight_id: ID of the insight to report on
            report_type: Type of report ("executive", "detailed", "technical")
            custom_sections: Custom sections to include

        Returns:
            str: Formatted report
        """
        # Retrieve insight
        insight = await self.get_insight_by_id(insight_id)
        if not insight:
            return f"Error: Insight with ID {insight_id} not found"

        # Determine sections based on report type
        sections = []

        if report_type == "executive":
            sections = ["narrative", "recommendations"]
        elif report_type == "detailed":
            sections = ["narrative", "trends", "anomalies", "segmentation", "recommendations"]
        elif report_type == "technical":
            sections = ["profile", "trends", "anomalies", "segmentation", "correlations", "recommendations"]

        # Add custom sections if provided
        if custom_sections:
            sections.extend([s for s in custom_sections if s not in sections])

        # Generate report
        report = f"# Data Insight Report: {insight['context']}\n\n"
        report += f"Generated: {insight['timestamp']}\n\n"

        # Add sections
        for section in sections:
            if section in insight:
                # Handle nested sections
                if isinstance(insight[section], dict) and "narrative" in insight[section]:
                    report += f"## {section.title()}\n\n"
                    report += f"{insight[section]['narrative']}\n\n"
                # Handle recommendation list
                elif section == "recommendations" and isinstance(insight[section], list):
                    report += f"## Recommendations\n\n"
                    for i, rec in enumerate(insight[section], 1):
                        report += f"### {i}. {rec.get('title')}\n\n"
                        report += f"{rec.get('description')}\n\n"
                        report += f"**Impact**: {rec.get('impact')}\n\n"
                        report += f"**Measurement**: {rec.get('measurement')}\n\n"
                else:
                    report += f"## {section.title()}\n\n"
                    report += f"{insight[section]}\n\n"

        return report


# Factory function to create a data insight agent with default configuration
def create_data_insight_agent(**kwargs) -> DataInsightAgent:
    """
    Create a Data Insight Agent with default configuration.

    Args:
        **kwargs: Override default configuration parameters

    Returns:
        DataInsightAgent: Configured agent instance
    """
    return DataInsightAgent(**kwargs)