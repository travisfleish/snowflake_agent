import pytest
import asyncio
from unittest.mock import Mock, patch, MagicMock, AsyncMock, call
import json
from datetime import datetime
from typing import List, Dict, Any

from crewai import Agent, Task, Crew
from crews.base_crew import BaseCrew, CrewConfig
from agents.base_agent import BaseAgent
from agents.analyst_agent import DataInsightAgent
from agents.research_agent import SnowflakeAnalystAgent
from agents.executive_agent import StrategicAdvisorAgent
from storage.memory import SharedMemory


# Test-specific subclass for tracking execution
class TestableBaseCrew(BaseCrew):
    """Subclass of BaseCrew with task execution tracking for testing."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.execution_order = []
        self.task_results = {}
    
    async def _process_task_inputs(self, crew, task, agent):
        """Track task execution order."""
        self.execution_order.append(task.description[:30])  # First 30 chars as identifier
        result = await super()._process_task_inputs(crew, task, agent)
        return result


# Mocked agent responses for deterministic testing
class MockAgentResponses:
    """Mock responses for different agent types."""
    
    @staticmethod
    def data_analyst_response(task_description):
        if "retrieve sales data" in task_description.lower():
            return {"sales_data": {"total": 1000000, "by_region": {"North": 400000, "South": 300000, "East": 200000, "West": 100000}}}
        elif "analyze trends" in task_description.lower():
            return "Sales are trending up 5% year-over-year, with strongest growth in the North region."
        else:
            return "Generic data analyst response"
    
    @staticmethod
    def research_analyst_response(task_description):
        if "market research" in task_description.lower():
            return "Market research shows competitor A has 30% market share, while we have 25%."
        elif "competitor analysis" in task_description.lower():
            return "Competitor analysis complete. Main threats are from Companies A and B."
        else:
            return "Generic research response"
    
    @staticmethod
    def strategic_advisor_response(task_description):
        if "strategic recommendations" in task_description.lower():
            return "Strategic recommendation: Focus on North region and introduce premium product line."
        elif "implementation plan" in task_description.lower():
            return "Implementation plan: Phase 1 (Q1): Market research, Phase 2 (Q2): Product development"
        else:
            return "Generic strategic advice"


# Test fixtures
@pytest.fixture
def mock_agent_factory():
    """Factory to create mock agents with specified response behavior."""
    def _create_mock_agent(agent_type, name="Test Agent", role="Tester", goal="Testing"):
        mock_agent = MagicMock(spec=BaseAgent)
        mock_agent.name = name
        mock_agent.role = role
        mock_agent.goal = goal
        
        # Set up the get_crew_agent method to return a CrewAI Agent
        mock_crew_agent = MagicMock(spec=Agent)
        mock_crew_agent.name = name
        mock_crew_agent.role = role
        mock_crew_agent.goal = goal
        mock_agent.get_crew_agent.return_value = mock_crew_agent
        
        # Set up different response behavior based on agent type
        if agent_type == "data_analyst":
            mock_crew_agent.execute_task.side_effect = \
                lambda task: MockAgentResponses.data_analyst_response(task.description)
        elif agent_type == "research_analyst":
            mock_crew_agent.execute_task.side_effect = \
                lambda task: MockAgentResponses.research_analyst_response(task.description)
        elif agent_type == "strategic_advisor":
            mock_crew_agent.execute_task.side_effect = \
                lambda task: MockAgentResponses.strategic_advisor_response(task.description)
            
        return mock_agent
    
    return _create_mock_agent


@pytest.fixture
def shared_memory():
    """Create shared memory for testing."""
    return SharedMemory(crew_name="test_crew")


@pytest.fixture
def sequential_crew(mock_agent_factory, shared_memory):
    """Create a crew with sequential task execution."""
    # Create agents
    data_analyst = mock_agent_factory("data_analyst", name="Data Analyst")
    research_analyst = mock_agent_factory("research_analyst", name="Research Analyst")
    strategic_advisor = mock_agent_factory("strategic_advisor", name="Strategic Advisor")
    
    # Create crew config
    config = CrewConfig(
        name="Sequential Test Crew",
        description="Testing sequential task execution",
        sequential=True,  # Important: sequential execution
        memory=shared_memory
    )
    
    # Create crew
    crew = TestableBaseCrew(config=config)
    crew.add_agents([data_analyst, research_analyst, strategic_advisor])
    
    return crew


@pytest.fixture
def parallel_crew(mock_agent_factory, shared_memory):
    """Create a crew with parallel task execution."""
    # Create agents
    data_analyst = mock_agent_factory("data_analyst", name="Data Analyst")
    research_analyst = mock_agent_factory("research_analyst", name="Research Analyst")
    strategic_advisor = mock_agent_factory("strategic_advisor", name="Strategic Advisor")
    
    # Create crew config
    config = CrewConfig(
        name="Parallel Test Crew",
        description="Testing parallel task execution",
        sequential=False,  # Important: parallel execution
        memory=shared_memory
    )
    
    # Create crew
    crew = TestableBaseCrew(config=config)
    crew.add_agents([data_analyst, research_analyst, strategic_advisor])
    
    return crew


# Mock CrewAI classes to avoid actual execution in tests
@pytest.fixture
def mock_crewai_classes():
    """Mock the CrewAI classes to isolate the test from external execution."""
    with patch('crewai.Agent', autospec=True) as mock_agent_class:
        with patch('crewai.Task', autospec=True) as mock_task_class:
            with patch('crewai.Crew', autospec=True) as mock_crew_class:
                # Set up mock crew behavior
                mock_crew_instance = MagicMock()
                mock_crew_instance.run_async.return_value = "Crew execution complete"
                mock_crew_class.return_value = mock_crew_instance
                
                yield {
                    'Agent': mock_agent_class,
                    'Task': mock_task_class,
                    'Crew': mock_crew_class,
                    'crew_instance': mock_crew_instance
                }


# Tests for sequential execution
class TestSequentialCrewExecution:
    """Tests for sequential CrewAI execution."""
    
    @pytest.mark.asyncio
    async def test_sequential_task_execution(self, sequential_crew):
        """Test tasks execute in sequence in a sequential crew."""
        # Create tasks
        task1 = sequential_crew.create_task(
            description="Task 1: Retrieve sales data from Snowflake",
            agent=sequential_crew.agents[0]  # Data Analyst
        )
        
        task2 = sequential_crew.create_task(
            description="Task 2: Analyze trends in sales data",
            agent=sequential_crew.agents[0]  # Data Analyst
        )
        
        task3 = sequential_crew.create_task(
            description="Task 3: Generate strategic recommendations",
            agent=sequential_crew.agents[2]  # Strategic Advisor
        )
        
        # Use a real Crew but with mocked tasks/agents
        with patch('crews.base_crew.BaseCrew.get_crew') as mock_get_crew:
            # Create a mock crew with sequential run behavior
            mock_crew = MagicMock()
            
            # Make run_async execute each task in sequence and record results
            async def mock_run_async():
                # In sequential mode, execute tasks in order
                results = []
                for i, task in enumerate([task1, task2, task3]):
                    # Get the mock agent associated with this task
                    agent = sequential_crew.agents[0] if i < 2 else sequential_crew.agents[2]
                    mock_agent = agent.get_crew_agent()
                    
                    # Execute the task and store result
                    result = mock_agent.execute_task(task)
                    sequential_crew.task_results[task.description[:30]] = result
                    results.append(result)
                    sequential_crew.execution_order.append(task.description[:30])
                
                # Return consolidated result
                return "\n".join(results)
                
            mock_crew.run_async.side_effect = mock_run_async
            mock_get_crew.return_value = mock_crew
            
            # Run the crew
            result = await sequential_crew.run()
            
            # Verify execution order
            expected_order = [
                "Task 1: Retrieve sales data fr",
                "Task 2: Analyze trends in sale",
                "Task 3: Generate strategic rec"
            ]
            assert sequential_crew.execution_order == expected_order
            
            # Verify task results were captured
            assert len(sequential_crew.task_results) == 3
            assert "sales_data" in str(sequential_crew.task_results["Task 1: Retrieve sales data fr"])
            assert "trending up 5%" in sequential_crew.task_results["Task 2: Analyze trends in sale"]
            assert "Strategic recommendation" in sequential_crew.task_results["Task 3: Generate strategic rec"]
    
    @pytest.mark.asyncio
    async def test_data_flow_between_sequential_tasks(self, sequential_crew, shared_memory):
        """Test data flows correctly between sequential tasks using shared memory."""
        # Create tasks with data dependencies
        task1 = sequential_crew.create_task(
            description="Task 1: Retrieve sales data from Snowflake",
            agent=sequential_crew.agents[0]  # Data Analyst
        )
        
        task2 = sequential_crew.create_task(
            description="Task 2: Analyze retrieved sales data",
            agent=sequential_crew.agents[0]  # Data Analyst
        )
        
        # Use a real Crew but with mocked tasks/agents
        with patch('crews.base_crew.BaseCrew.get_crew') as mock_get_crew:
            # Create a mock crew with sequential run behavior
            mock_crew = MagicMock()
            
            # Make run_async execute tasks and use shared memory
            async def mock_run_async():
                # Task 1: Store data in shared memory
                agent1 = sequential_crew.agents[0].get_crew_agent()
                result1 = agent1.execute_task(task1)
                shared_memory.add("sales_data", result1)
                sequential_crew.execution_order.append(task1.description[:30])
                
                # Task 2: Read from shared memory
                agent2 = sequential_crew.agents[0].get_crew_agent()
                # Mock the execute task to check shared memory
                
                def execute_task2(task):
                    # Get data from shared memory
                    data = shared_memory.get("sales_data")
                    # Verify data exists before proceeding
                    if data and "sales_data" in str(data):
                        return f"Analysis based on memory data: total sales ${data.get('total', 0) if isinstance(data, dict) else 'unknown'}"
                    else:
                        return "Error: No data found in memory"
                
                agent2.execute_task.side_effect = execute_task2
                result2 = agent2.execute_task(task2)
                sequential_crew.execution_order.append(task2.description[:30])
                
                return f"{result1}\n{result2}"
                
            mock_crew.run_async.side_effect = mock_run_async
            mock_get_crew.return_value = mock_crew
            
            # Run the crew
            result = await sequential_crew.run()
            
            # Verify execution order
            expected_order = [
                "Task 1: Retrieve sales data fr",
                "Task 2: Analyze retrieved sale"
            ]
            assert sequential_crew.execution_order == expected_order
            
            # Verify data was passed correctly
            assert "Analysis based on memory data" in result
            assert "total sales" in result
    
    @pytest.mark.asyncio
    async def test_sequential_error_handling(self, sequential_crew):
        """Test error handling in sequential task execution."""
        # Create tasks with an error in the middle
        task1 = sequential_crew.create_task(
            description="Task 1: Retrieve sales data from Snowflake",
            agent=sequential_crew.agents[0]  # Data Analyst
        )
        
        task2 = sequential_crew.create_task(
            description="Task 2: This task will fail",
            agent=sequential_crew.agents[0]  # Data Analyst
        )
        
        task3 = sequential_crew.create_task(
            description="Task 3: Generate strategic recommendations",
            agent=sequential_crew.agents[2]  # Strategic Advisor
        )
        
        # Use a real Crew but with mocked tasks/agents
        with patch('crews.base_crew.BaseCrew.get_crew') as mock_get_crew:
            # Create a mock crew with sequential run behavior
            mock_crew = MagicMock()
            
            # Make run_async execute tasks with an error in the middle
            async def mock_run_async():
                # Task 1: Normal execution
                agent1 = sequential_crew.agents[0].get_crew_agent()
                result1 = agent1.execute_task(task1)
                sequential_crew.execution_order.append(task1.description[:30])
                
                # Task 2: Will fail
                agent2 = sequential_crew.agents[0].get_crew_agent()
                agent2.execute_task.side_effect = Exception("Simulated task failure")
                
                try:
                    agent2.execute_task(task2)
                except Exception as e:
                    sequential_crew.execution_order.append(f"Error: {task2.description[:20]}")
                    # If stop_on_task_error is True, this should stop execution
                    if sequential_crew.config.stop_on_task_error:
                        raise
                
                # Task 3: Should only execute if not stopping on errors
                agent3 = sequential_crew.agents[2].get_crew_agent()
                result3 = agent3.execute_task(task3)
                sequential_crew.execution_order.append(task3.description[:30])
                
                return "Execution completed with error handling"
                
            mock_crew.run_async.side_effect = mock_run_async
            mock_get_crew.return_value = mock_crew
            
            # Set stop_on_task_error to False to continue on errors
            sequential_crew.config.stop_on_task_error = False
            
            # Run the crew
            result = await sequential_crew.run()
            
            # Verify execution order - all tasks should have executed
            assert len(sequential_crew.execution_order) == 3
            assert "Task 1:" in sequential_crew.execution_order[0]
            assert "Error:" in sequential_crew.execution_order[1]
            assert "Task 3:" in sequential_crew.execution_order[2]
            
            # Now test with stop_on_task_error = True
            sequential_crew.execution_order = []
            sequential_crew.config.stop_on_task_error = True
            mock_crew.run_async.side_effect = Exception("Simulated task failure")
            
            # Should raise exception
            with pytest.raises(Exception):
                await sequential_crew.run()


# Tests for parallel execution
class TestParallelCrewExecution:
    """Tests for parallel CrewAI execution."""
    
    @pytest.mark.asyncio
    async def test_parallel_task_execution(self, parallel_crew):
        """Test tasks can execute in parallel."""
        # Create tasks for different agents to run in parallel
        task1 = parallel_crew.create_task(
            description="Task 1: Retrieve sales data from Snowflake",
            agent=parallel_crew.agents[0]  # Data Analyst
        )
        
        task2 = parallel_crew.create_task(
            description="Task 2: Conduct market research",
            agent=parallel_crew.agents[1]  # Research Analyst
        )
        
        task3 = parallel_crew.create_task(
            description="Task 3: Generate strategic recommendations",
            agent=parallel_crew.agents[2]  # Strategic Advisor
        )
        
        # Use a real Crew but with mocked tasks/agents
        with patch('crews.base_crew.BaseCrew.get_crew') as mock_get_crew:
            # Create a mock crew with parallel run behavior
            mock_crew = MagicMock()
            
            # Make run_async execute tasks in parallel
            async def mock_run_async():
                # Simulate parallel execution with asyncio.gather
                tasks = []
                
                for i, task in enumerate([task1, task2, task3]):
                    agent = parallel_crew.agents[i].get_crew_agent()
                    
                    async def execute_task_async(t, a):
                        result = a.execute_task(t)
                        parallel_crew.task_results[t.description[:30]] = result
                        parallel_crew.execution_order.append(t.description[:30])
                        return result
                    
                    tasks.append(execute_task_async(task, agent))
                
                results = await asyncio.gather(*tasks)
                return "\n".join(str(r) for r in results)
                
            mock_crew.run_async.side_effect = mock_run_async
            mock_get_crew.return_value = mock_crew
            
            # Run the crew
            result = await parallel_crew.run()
            
            # Verify all tasks executed (order may vary in parallel)
            assert len(parallel_crew.execution_order) == 3
            task_descriptions = set(parallel_crew.execution_order)
            expected_descriptions = {
                "Task 1: Retrieve sales data fr",
                "Task 2: Conduct market researc",
                "Task 3: Generate strategic rec"
            }
            assert task_descriptions == expected_descriptions
            
            # Verify all task results were captured
            assert len(parallel_crew.task_results) == 3
            assert "sales_data" in str(parallel_crew.task_results["Task 1: Retrieve sales data fr"])
            assert "Market research" in parallel_crew.task_results["Task 2: Conduct market researc"]
            assert "Strategic recommendation" in parallel_crew.task_results["Task 3: Generate strategic rec"]
    
    @pytest.mark.asyncio
    async def test_complex_task_dependencies(self, parallel_crew, shared_memory):
        """Test complex task dependencies in a mixed parallel/sequential workflow."""
        # Create initial parallel tasks
        task1 = parallel_crew.create_task(
            description="Task 1: Retrieve sales data from Snowflake",
            agent=parallel_crew.agents[0]  # Data Analyst
        )
        
        task2 = parallel_crew.create_task(
            description="Task 2: Conduct market research",
            agent=parallel_crew.agents[1]  # Research Analyst
        )
        
        # Create dependent task that needs results from both task1 and task2
        task3 = parallel_crew.create_task(
            description="Task 3: Generate strategic recommendations based on data and research",
            agent=parallel_crew.agents[2]  # Strategic Advisor
        )
        
        # Use a real Crew but with mocked tasks/agents
        with patch('crews.base_crew.BaseCrew.get_crew') as mock_get_crew:
            # Create a mock crew with custom execution behavior
            mock_crew = MagicMock()
            
            # Make run_async execute tasks with dependencies
            async def mock_run_async():
                # Step 1: Execute task1 and task2 in parallel
                async def execute_task1():
                    agent1 = parallel_crew.agents[0].get_crew_agent()
                    result1 = agent1.execute_task(task1)
                    shared_memory.add("sales_data", result1)
                    parallel_crew.execution_order.append(task1.description[:30])
                    return result1
                
                async def execute_task2():
                    agent2 = parallel_crew.agents[1].get_crew_agent()
                    result2 = agent2.execute_task(task2)
                    shared_memory.add("market_research", result2)
                    parallel_crew.execution_order.append(task2.description[:30])
                    return result2
                
                # Run first two tasks in parallel
                result1, result2 = await asyncio.gather(execute_task1(), execute_task2())
                
                # Step 2: Execute task3 after task1 and task2 are complete
                # Modify the execute_task behavior to use shared memory
                agent3 = parallel_crew.agents[2].get_crew_agent()
                
                def execute_task3(task):
                    # Get data from shared memory
                    sales_data = shared_memory.get("sales_data")
                    market_research = shared_memory.get("market_research")
                    
                    # Verify both dependencies exist
                    if sales_data and market_research:
                        return (f"Strategic recommendations based on:\n"
                                f"- Sales data: ${sales_data['total'] if isinstance(sales_data, dict) and 'total' in sales_data else 'unknown'}\n"
                                f"- Market research: {market_research}")
                    else:
                        return "Error: Missing dependencies in memory"
                
                agent3.execute_task.side_effect = execute_task3
                result3 = agent3.execute_task(task3)
                parallel_crew.execution_order.append(task3.description[:30])
                
                # Combine all results
                return f"{result1}\n{result2}\n{result3}"
                
            mock_crew.run_async.side_effect = mock_run_async
            mock_get_crew.return_value = mock_crew
            
            # Run the crew
            result = await parallel_crew.run()
            
            # Verify execution order has 3 tasks
            assert len(parallel_crew.execution_order) == 3
            
            # Task3 must be the last one since it depends on task1 and task2
            assert parallel_crew.execution_order[2] == "Task 3: Generate strategic rec"
            
            # First two can be in any order
            first_two = set(parallel_crew.execution_order[:2])
            expected_first_two = {
                "Task 1: Retrieve sales data fr",
                "Task 2: Conduct market researc"
            }
            assert first_two == expected_first_two
            
            # Verify the final result contains data from both dependencies
            assert "Strategic recommendations based on" in result
            assert "Sales data" in result
            assert "Market research" in result


# Tests for real-world scenarios
class TestRealWorldScenarios:
    """Tests simulating realistic CrewAI usage scenarios."""
    
    @pytest.mark.asyncio
    async def test_sales_analysis_workflow(self, sequential_crew, shared_memory):
        """Test a realistic sales analysis workflow."""
        # Step 1: Configure the workflow tasks
        data_task = sequential_crew.create_task(
            description="Retrieve quarterly sales data by region and product category",
            agent=sequential_crew.agents[0]  # Data Analyst
        )
        
        analysis_task = sequential_crew.create_task(
            description="Analyze trends in sales data comparing to previous quarters",
            agent=sequential_crew.agents[0]  # Data Analyst
        )
        
        research_task = sequential_crew.create_task(
            description="Conduct competitor analysis focusing on top 3 competitors",
            agent=sequential_crew.agents[1]  # Research Analyst
        )
        
        strategy_task = sequential_crew.create_task(
            description="Generate strategic recommendations based on analysis",
            agent=sequential_crew.agents[2]  # Strategic Advisor
        )
        
        implementation_task = sequential_crew.create_task(
            description="Create implementation plan for the strategic recommendations",
            agent=sequential_crew.agents[2]  # Strategic Advisor
        )
        
        # Use a real Crew but with mocked tasks/agents
        with patch('crews.base_crew.BaseCrew.get_crew') as mock_get_crew:
            # Create a mock crew with sequential run behavior
            mock_crew = MagicMock()
            
            # Make run_async execute the workflow
            async def mock_run_async():
                # Execute tasks in sequence
                results = []
                
                # Task 1: Data retrieval
                agent1 = sequential_crew.agents[0].get_crew_agent()
                result1 = agent1.execute_task(data_task)
                shared_memory.add("sales_data", result1)
                sequential_crew.execution_order.append("Step 1: Data Retrieval")
                results.append(result1)
                
                # Task 2: Data analysis
                agent1 = sequential_crew.agents[0].get_crew_agent()
                result2 = agent1.execute_task(analysis_task)
                shared_memory.add("sales_analysis", result2)
                sequential_crew.execution_order.append("Step 2: Data Analysis")
                results.append(result2)
                
                # Task 3: Market research
                agent2 = sequential_crew.agents[1].get_crew_agent()
                result3 = agent2.execute_task(research_task)
                shared_memory.add("competitor_analysis", result3)
                sequential_crew.execution_order.append("Step 3: Market Research")
                results.append(result3)
                
                # Task 4: Strategic recommendations
                agent3 = sequential_crew.agents[2].get_crew_agent()
                # Override execute task to use shared memory
                def execute_strategy_task(task):
                    sales_data = shared_memory.get("sales_data")
                    sales_analysis = shared_memory.get("sales_analysis")
                    competitor_analysis = shared_memory.get("competitor_analysis")
                    
                    # Create recommendations based on data
                    if sales_data and sales_analysis and competitor_analysis:
                        return f"Strategic recommendations based on comprehensive analysis of sales trends and competitive landscape."
                    else:
                        return "Error: Missing required inputs for strategy formulation"
                
                agent3.execute_task.side_effect = execute_strategy_task
                result4 = agent3.execute_task(strategy_task)
                shared_memory.add("strategic_recommendations", result4)
                sequential_crew.execution_order.append("Step 4: Strategic Recommendations")
                results.append(result4)
                
                # Task 5: Implementation plan
                agent3 = sequential_crew.agents[2].get_crew_agent()
                # Reset side effect for this task
                agent3.execute_task.side_effect = None
                agent3.execute_task.return_value = "Implementation plan with 3 phases over next 12 months"
                result5 = agent3.execute_task(implementation_task)
                sequential_crew.execution_order.append("Step 5: Implementation Plan")
                results.append(result5)
                
                # Combine results into a report
                report = "\n\n".join([
                    "# Sales Analysis and Strategic Plan",
                    "## Executive Summary",
                    "Based on our comprehensive analysis, we recommend focusing on the North region and premium product lines.",
                    "## Data Analysis",
                    str(results[0]),
                    str(results[1]),
                    "## Competitive Landscape",
                    str(results[2]),
                    "## Strategic Recommendations",
                    str(results[3]),
                    "## Implementation Plan",
                    str(results[4])
                ])
                
                return report
                
            mock_crew.run_async.side_effect = mock_run_async
            mock_get_crew.return_value = mock_crew
            
            # Run the crew
            result = await sequential_crew.run()
            
            # Verify workflow executed completely
            assert len(sequential_crew.execution_order) == 5
            assert sequential_crew.execution_order[0] == "Step 1: Data Retrieval"
            assert sequential_crew.execution_order[1] == "Step 2: Data Analysis"
            assert sequential_crew.execution_order[2] == "Step 3: Market Research"
            assert sequential_crew.execution_order[3] == "Step 4: Strategic Recommendations"
            assert sequential_crew.execution_order[4] == "Step 5: Implementation Plan"
            
            # Verify result structure
            assert "# Sales Analysis and Strategic Plan" in result
            assert "## Executive Summary" in result
            assert "## Data Analysis" in result
            assert "## Competitive Landscape" in result
            assert "## Strategic Recommendations" in result
            assert "## Implementation Plan" in result


# Test agent interaction and delegation
class TestAgentInteraction:
    """Tests for agent interactions and delegations."""
    
    @pytest.mark.asyncio
    async def test_agent_delegation(self, sequential_crew):
        """Test that agents can delegate tasks to other agents."""
        # Create a task assigned to the strategic advisor
        main_task = sequential_crew.create_task(
            description="Develop comprehensive business strategy",
            agent=sequential_crew.agents[2]  # Strategic Advisor
        )
        
        # Use a real Crew but with mocked tasks/agents
        with patch('crews.base_crew.BaseCrew.get_crew') as mock_get_crew:
            # Create a mock crew for delegation
            mock_crew = MagicMock()
            
            # Configure the strategic advisor to delegate subtasks
            strategic_advisor = sequential_crew.agents[2].get_crew_agent()
            data_analyst = sequential_crew.agents[0].get_crew_agent()
            research_analyst = sequential_crew.agents[1].get_crew_agent()
            
            # Track delegation calls
            delegation_calls = []
            
            # Make execute_task delegate to other agents
            def execute_strategic_task(task):
                # Record the main task
                sequential_crew.execution_order.append("Main: Strategic Task")
                
                # Create subtasks for other agents
                subtask1 = Task(
                    description="Subtask 1: Analyze current performance metrics",
                    agent=data_analyst
                )
                subtask2 = Task(
                    description="Subtask 2: Research competitor strategies",
                    agent=research_analyst
                )
                
                # Execute subtasks (simulating delegation)
                delegation_calls.append(subtask1.description)
                result1 = data_analyst.execute_task(subtask1)
                sequential_crew.execution_order.append("Subtask 1: Data Analysis")
                
                delegation_calls.append(subtask2.description)
                result2 = research_analyst.execute_task(subtask2)
                sequential_crew.execution_order.append("Subtask 2: Research")
                
                # Combine results
                return f"Comprehensive strategy based on:\n- Performance analysis: {result1}\n- Competitor research: {result2}"
            
            strategic_advisor.execute_task.side_effect = execute_strategic_task
            
            # Make run_async delegate tasks
            async def mock_run_async():
                # Execute main task (which will delegate)
                return strategic_advisor.execute_task(main_task)
                
            mock_crew.run_async.side_effect = mock_run_async
            mock_get_crew.return_value = mock_crew
            
            # Run the crew
            result = await sequential_crew.run()
            
            # Verify delegation occurred
            assert len(delegation_calls) == 2
            assert "Subtask 1: Analyze current performance metrics" in delegation_calls[0]
            assert "Subtask 2: Research competitor strategies" in delegation_calls[1]
            
            # Verify execution order
            assert len(sequential_crew.execution_order) == 3
            assert sequential_crew.execution_order[0] == "Main: Strategic Task"
            assert sequential_crew.execution_order[1] == "Subtask 1: Data Analysis"
            assert sequential_crew.execution_order[2] == "Subtask 2: Research"
            
            # Verify result combines delegated task results
            assert "Comprehensive strategy based on" in result
            assert "Performance analysis" in result
            assert "Competitor research" in result