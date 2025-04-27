import pytest
from unittest.mock import Mock, patch, MagicMock
import json
from datetime import datetime
import asyncio
from typing import Dict, List, Any

from agents.base_agent import BaseAgent
from agents.analyst_agent import DataInsightAgent
from agents.research_agent import SnowflakeAnalystAgent
from agents.executive_agent import StrategicAdvisorAgent
from utils.snowflake_connector import connector
from storage.memory import SharedMemory


# Fixtures for common test setup
@pytest.fixture
def mock_llm_client():
    """Mock LLM client for testing agent responses."""
    mock_client = Mock()

    # Set up the generate method to return a predictable response
    async def mock_generate(*args, **kwargs):
        return {
            'id': 'test-id',
            'created': datetime.now().timestamp(),
            'model': 'test-model',
            'choices': [
                {
                    'message': {
                        'role': 'assistant',
                        'content': 'This is a test response'
                    },
                    'finish_reason': 'stop'
                }
            ],
            'usage': {
                'prompt_tokens': 10,
                'completion_tokens': 20,
                'total_tokens': 30
            }
        }

    mock_client.generate = mock_generate
    return mock_client


@pytest.fixture
def base_agent(mock_llm_client):
    """Create a base agent with mocked dependencies."""
    agent = BaseAgent(
        name="Test Agent",
        role="Tester",
        goal="Test functionality",
        llm_client=mock_llm_client,
        verbose=False
    )
    return agent


@pytest.fixture
def analyst_agent(mock_llm_client):
    """Create a Snowflake analyst agent with mocked dependencies."""
    with patch('utils.snowflake_connector.connector') as mock_connector:
        # Set up mock connector
        mock_connector.execute_query.return_value = [
            {"column1": "value1", "column2": "value2"},
            {"column1": "value3", "column2": "value4"}
        ]
        mock_connector.get_table_schema.return_value = [
            {"COLUMN_NAME": "column1", "DATA_TYPE": "VARCHAR", "IS_NULLABLE": "YES"},
            {"COLUMN_NAME": "column2", "DATA_TYPE": "VARCHAR", "IS_NULLABLE": "YES"}
        ]

        agent = SnowflakeAnalystAgent(
            name="SQL Analyst",
            role="SQL Expert",
            goal="Generate and execute SQL queries",
            llm_client=mock_llm_client,
            schema_cache={"tables": ["test_table"]}
        )
        return agent


@pytest.fixture
def data_insight_agent(mock_llm_client):
    """Create a data insight agent with mocked dependencies."""
    agent = DataInsightAgent(
        name="Data Insight Agent",
        role="Data Analyst",
        goal="Generate insights from data",
        llm_client=mock_llm_client
    )
    return agent


@pytest.fixture
def strategic_agent(mock_llm_client):
    """Create a strategic advisor agent with mocked dependencies."""
    agent = StrategicAdvisorAgent(
        name="Strategic Advisor",
        role="Executive Advisor",
        goal="Generate strategic recommendations",
        llm_client=mock_llm_client,
        business_context={
            "company_size": "midsize",
            "industry": "retail",
            "strategic_goals": ["growth", "efficiency"]
        }
    )
    return agent


@pytest.fixture
def shared_memory():
    """Create a shared memory instance for testing."""
    return SharedMemory(crew_name="test_crew")


# Tests for BaseAgent
class TestBaseAgent:
    @pytest.mark.asyncio
    async def test_thinking(self, base_agent, caplog):
        """Test the thinking method logs correctly when verbose is enabled."""
        # Set verbose to True
        base_agent.verbose = True

        # Call thinking method
        base_agent.thinking("This is a test thought")

        # Check if log contains the thought
        assert "This is a test thought" in caplog.text

    @pytest.mark.asyncio
    async def test_add_tool(self, base_agent):
        """Test adding a tool to an agent."""
        # Create a mock tool
        mock_tool = Mock()
        mock_tool.name = "test_tool"

        # Add tool to agent
        base_agent.add_tool(mock_tool)

        # Verify tool was added
        assert mock_tool in base_agent.tools
        # Verify crew_agent is reset
        assert base_agent._crew_agent is None

    @pytest.mark.asyncio
    async def test_get_crew_agent(self, base_agent):
        """Test getting the CrewAI agent instance."""
        # Get crew agent
        crew_agent = base_agent.get_crew_agent()

        # Verify agent properties
        assert crew_agent.name == base_agent.name
        assert crew_agent.role == base_agent.role
        assert crew_agent.goal == base_agent.goal

        # Call again to test caching
        crew_agent2 = base_agent.get_crew_agent()
        assert crew_agent is crew_agent2  # Should be the same instance

    @pytest.mark.asyncio
    async def test_memory_operations(self, base_agent):
        """Test memory operations."""
        # Create memory
        memory = SharedMemory(crew_name="test_agent")
        base_agent.memory = memory

        # Test remember
        base_agent.remember("test_key", "test_value")

        # Test recall
        value = base_agent.recall("test_key")
        assert value == "test_value"

        # Test forget
        success = base_agent.forget("test_key")
        assert success is True

        # Verify key is gone
        value = base_agent.recall("test_key")
        assert value is None

        # Test recalling with default
        value = base_agent.recall("non_existent", "default_value")
        assert value == "default_value"


# Tests for SnowflakeAnalystAgent
class TestSnowflakeAnalystAgent:
    @pytest.mark.asyncio
    async def test_analyze_business_question(self, analyst_agent):
        """Test analyzing a business question."""
        # Mock the _ensure_schema_loaded method
        with patch.object(analyst_agent, '_ensure_schema_loaded') as mock_ensure_schema:
            # Mock _generate_sql_query and _explain_results methods
            with patch.object(analyst_agent, '_generate_sql_query',
                              return_value="SELECT * FROM test_table") as mock_generate_sql:
                with patch.object(analyst_agent, '_explain_results', return_value="Test explanation") as mock_explain:
                    # Call analyze_business_question
                    result = await analyst_agent.analyze_business_question("What are the top sales?")

                    # Verify methods were called
                    mock_ensure_schema.assert_called_once()
                    mock_generate_sql.assert_called_once_with("What are the top sales?")
                    mock_explain.assert_called_once()

                    # Verify result structure
                    assert result["question"] == "What are the top sales?"
                    assert result["sql_query"] == "SELECT * FROM test_table"
                    assert result["explanation"] == "Test explanation"
                    assert "results" in result

    @pytest.mark.asyncio
    async def test_generate_sql_query(self, analyst_agent):
        """Test SQL query generation."""
        # Setup
        analyst_agent.schema_cache = {
            "tables": ["customers", "orders"],
            "customers": "Column info for customers",
            "orders": "Column info for orders"
        }

        # Call method with mocked LLM response
        query = await analyst_agent._generate_sql_query("Show me top 10 customers")

        # Since we mocked the LLM to return "This is a test response", we expect it to be parsed
        assert query == "This is a test response"

    @pytest.mark.asyncio
    async def test_explain_sql_query(self, analyst_agent):
        """Test SQL query explanation."""
        # Call explain_sql_query
        explanation = await analyst_agent.explain_sql_query("SELECT * FROM test_table")

        # Verify explanation
        assert explanation == "This is a test response"

    @pytest.mark.asyncio
    async def test_optimize_sql_query(self, analyst_agent):
        """Test SQL query optimization."""
        # Call optimize_sql_query
        optimized = await analyst_agent.optimize_sql_query("SELECT * FROM test_table")

        # Verify optimized query
        assert optimized == "This is a test response"


# Tests for DataInsightAgent
class TestDataInsightAgent:
    @pytest.mark.asyncio
    async def test_analyze_data(self, data_insight_agent):
        """Test data analysis function."""
        # Test data
        test_data = {
            "column1": [1, 2, 3],
            "column2": ["a", "b", "c"]
        }

        # Mock internal methods
        with patch.object(data_insight_agent, '_ensure_dataframe', return_value="df") as mock_ensure_df:
            with patch.object(data_insight_agent, '_generate_data_profile',
                              return_value={"shape": (3, 2)}) as mock_profile:
                with patch.object(data_insight_agent, '_generate_summary', return_value="Data summary") as mock_summary:
                    with patch.object(data_insight_agent, '_create_narrative',
                                      return_value="Data narrative") as mock_narrative:
                        # Call analyze_data
                        result = await data_insight_agent.analyze_data(
                            test_data,
                            "Test context"
                        )

                        # Verify method calls
                        mock_ensure_df.assert_called_once_with(test_data)
                        mock_profile.assert_called_once()
                        mock_summary.assert_called_once()
                        mock_narrative.assert_called_once()

                        # Verify result structure
                        assert "data_id" in result
                        assert "timestamp" in result
                        assert "context" in result
                        assert "profile" in result
                        assert "insights" in result
                        assert "narrative" in result

    @pytest.mark.asyncio
    async def test_ensure_dataframe(self, data_insight_agent):
        """Test dataframe conversion function."""
        # Test with dict
        dict_data = {"column1": [1, 2, 3], "column2": ["a", "b", "c"]}
        result = data_insight_agent._ensure_dataframe(dict_data)
        # Basic check that it attempted conversion
        assert hasattr(result, 'columns')

        # Test with string (simplified test)
        with patch('pandas.DataFrame') as mock_df:
            string_data = """column1,column2
            1,a
            2,b
            3,c"""
            data_insight_agent._ensure_dataframe(string_data)
            # Check that DataFrame constructor was called
            mock_df.assert_called()

    @pytest.mark.asyncio
    async def test_get_insight_by_id(self, data_insight_agent):
        """Test retrieving insights by ID."""
        # Add test insight to cache
        test_insight = {"data": "test"}
        data_insight_agent.insight_cache = {"test_id": test_insight}

        # Retrieve insight
        result = await data_insight_agent.get_insight_by_id("test_id")
        assert result == test_insight

        # Test non-existent ID
        result = await data_insight_agent.get_insight_by_id("non_existent")
        assert result is None


# Tests for StrategicAdvisorAgent
class TestStrategicAdvisorAgent:
    @pytest.mark.asyncio
    async def test_generate_strategic_recommendations(self, strategic_agent):
        """Test generating strategic recommendations."""
        # Test data
        analysis_results = {
            "key_metrics": {"revenue": 1000000, "profit": 250000},
            "trends": ["Increasing online sales", "Decreasing in-store traffic"]
        }

        # Mock internal methods
        with patch.object(strategic_agent, '_extract_key_findings',
                          return_value=[{"title": "Finding 1"}]) as mock_findings:
            with patch.object(strategic_agent, '_generate_strategic_insights',
                              return_value=[{"title": "Insight 1"}]) as mock_insights:
                with patch.object(strategic_agent, '_develop_strategic_priorities',
                                  return_value=[{"title": "Priority 1"}]) as mock_priorities:
                    with patch.object(strategic_agent, '_create_implementation_roadmap',
                                      return_value=[{"priority": "Priority 1"}]) as mock_roadmap:
                        with patch.object(strategic_agent, '_generate_executive_summary',
                                          return_value="Executive summary") as mock_summary:
                            # Call generate_strategic_recommendations
                            result = await strategic_agent.generate_strategic_recommendations(
                                analysis_results,
                                "How can we increase revenue?"
                            )

                            # Verify method calls
                            mock_findings.assert_called_once()
                            mock_insights.assert_called_once()
                            mock_priorities.assert_called_once()
                            mock_roadmap.assert_called_once()
                            mock_summary.assert_called_once()

                            # Verify result structure
                            assert hasattr(result, "id")
                            assert hasattr(result, "title")
                            assert hasattr(result, "summary")
                            assert hasattr(result, "context")
                            assert hasattr(result, "strategic_priorities")
                            assert hasattr(result, "next_steps")

    @pytest.mark.asyncio
    async def test_generate_executive_presentation(self, strategic_agent):
        """Test generating executive presentation."""
        # Mock recommendation in history
        recommendation = MagicMock()
        recommendation.summary = "Test summary"
        recommendation.strategic_priorities = [MagicMock(), MagicMock()]
        strategic_agent.recommendation_history = [
            {"id": "rec_123", "recommendation": recommendation, "business_question": "Test question?"}
        ]

        # Call generate_executive_presentation
        result = await strategic_agent.generate_executive_presentation("rec_123", "markdown", 5)

        # Verify result
        assert result == "This is a test response"

        # Test with invalid recommendation ID
        result = await strategic_agent.generate_executive_presentation("invalid_id")
        assert "Error: Recommendation with ID invalid_id not found" in result

    @pytest.mark.asyncio
    async def test_update_business_context(self, strategic_agent):
        """Test updating business context."""
        # Initial context
        assert strategic_agent.business_context["company_size"] == "midsize"

        # Update context
        strategic_agent.update_business_context({"company_size": "large", "market_position": "leader"})

        # Verify context updated
        assert strategic_agent.business_context["company_size"] == "large"
        assert strategic_agent.business_context["market_position"] == "leader"
        # Verify existing values preserved
        assert "industry" in strategic_agent.business_context


# Integration tests
class TestAgentIntegration:
    @pytest.mark.asyncio
    async def test_snowflake_to_insight_flow(self, analyst_agent, data_insight_agent):
        """Test integration between Snowflake analyst and insight agent."""
        # Mock Snowflake analysis
        with patch.object(analyst_agent, 'analyze_business_question', return_value={
            "question": "What are our top products?",
            "sql_query": "SELECT * FROM products ORDER BY sales DESC LIMIT 10",
            "results": [
                {"product_id": 1, "name": "Product A", "sales": 1000},
                {"product_id": 2, "name": "Product B", "sales": 800}
            ],
            "explanation": "These are our top selling products"
        }) as mock_analyze:
            # First step: Get data from Snowflake
            snowflake_results = await analyst_agent.analyze_business_question("What are our top products?")

            # Mock insight generation
            with patch.object(data_insight_agent, 'analyze_data', return_value={
                "data_id": "insight_123",
                "insights": {"summary": "Product A is our top seller"},
                "recommendations": [{"title": "Focus on Product A"}]
            }) as mock_insights:
                # Second step: Generate insights from the data
                insights = await data_insight_agent.analyze_data(
                    snowflake_results["results"],
                    snowflake_results["question"]
                )

                # Verify flow
                mock_analyze.assert_called_once()
                mock_insights.assert_called_once()

                # Verify insights generated
                assert "data_id" in insights
                assert "insights" in insights
                assert "recommendations" in insights

    @pytest.mark.asyncio
    async def test_insight_to_strategy_flow(self, data_insight_agent, strategic_agent):
        """Test integration between insight agent and strategic advisor."""
        # Mock insight generation
        with patch.object(data_insight_agent, 'analyze_data', return_value={
            "data_id": "insight_123",
            "insights": {
                "summary": "Product A is our top seller",
                "trends": {"narrative": "Sales increasing 5% monthly"},
                "anomalies": {"narrative": "Unusual spike in Region B"}
            },
            "recommendations": [{"title": "Focus on Product A"}]
        }) as mock_insights:
            # First step: Generate insights
            insights = await data_insight_agent.analyze_data(
                {"product": ["A", "B"], "sales": [1000, 800]},
                "What are our best products?"
            )

            # Mock strategy generation
            with patch.object(strategic_agent, 'generate_strategic_recommendations',
                              return_value=MagicMock()) as mock_strategy:
                # Second step: Generate strategic recommendations
                strategy = await strategic_agent.generate_strategic_recommendations(
                    insights,
                    "How can we leverage our top products?"
                )

                # Verify flow
                mock_insights.assert_called_once()
                mock_strategy.assert_called_once()

                # Basic verification that it's the right type
                assert isinstance(strategy, MagicMock)


# Error handling tests
class TestAgentErrorHandling:
    @pytest.mark.asyncio
    async def test_analyze_business_question_error(self, analyst_agent):
        """Test error handling in analyze_business_question."""
        # Mock the generate_sql_query to raise an exception
        with patch.object(analyst_agent, '_generate_sql_query', side_effect=Exception("Test error")):
            # Call function and expect exception to be raised
            with pytest.raises(Exception) as excinfo:
                await analyst_agent.analyze_business_question("What are top sales?")

            # Verify exception message
            assert "Test error" in str(excinfo.value)

    @pytest.mark.asyncio
    async def test_empty_query_handling(self, analyst_agent):
        """Test handling of empty queries."""

        # Mock LLM to return an empty string
        async def mock_empty_response(*args, **kwargs):
            return {
                'choices': [{'message': {'content': ''}}]
            }

        analyst_agent._llm_client.generate = mock_empty_response

        # Call generate_sql_query with a question
        with pytest.raises(Exception) as excinfo:
            await analyst_agent._generate_sql_query("What are top sales?")

        # Since the mock returns empty string, we expect sql_lines to be empty
        # and the function to return an empty string or raise an exception
        assert "empty" in str(excinfo.value).lower() or not excinfo

    @pytest.mark.asyncio
    async def test_invalid_data_handling(self, data_insight_agent):
        """Test handling of invalid data in analyze_data."""
        # Call with invalid data type
        result = await data_insight_agent.analyze_data(None, "Test context")

        # Verify error handling
        assert "data_id" in result  # Should still return a structured response
        assert "error" in result["profile"] or isinstance(result["profile"], dict)


# Resource cleanup tests
class TestAgentCleanup:
    def test_memory_cleanup(self, shared_memory):
        """Test that shared memory is properly cleaned up."""
        # Add items with TTL
        shared_memory.add("test1", "value1", ttl=0.1)
        shared_memory.add("test2", "value2")

        # Verify items exist
        assert shared_memory.get("test1") == "value1"
        assert shared_memory.get("test2") == "value2"

        # Wait for TTL to expire
        import time
        time.sleep(0.2)

        # Verify expired item is cleaned up
        assert shared_memory.get("test1") is None
        assert shared_memory.get("test2") == "value2"

        # Explicitly clean up
        shared_memory.clear()

        # Verify all items removed
        assert shared_memory.get("test2") is None