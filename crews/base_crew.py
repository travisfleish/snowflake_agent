"""
Base crew implementation for Snowflake Agent system.
Provides foundational functionality for all crew types.
"""

import logging
from typing import List, Dict, Any, Optional, Union, Type, Callable
import datetime

from crewai import Crew, Agent, Task
from pydantic import BaseModel, Field

from agents.base_agent import BaseAgent
from config.settings import settings
from utils.prompt_templates import CrewPromptTemplates
from storage.memory import SharedMemory

# Configure logger
logger = logging.getLogger(__name__)


class CrewConfig(BaseModel):
    """Configuration for a crew."""

    name: str = Field(..., description="Name of the crew")
    description: str = Field(..., description="Description of the crew's purpose")
    verbose: bool = Field(False, description="Whether to enable verbose output")
    sequential: bool = Field(True, description="Whether to run tasks sequentially")
    memory: Optional[Any] = Field(None, description="Shared memory instance")
    max_rpm: Optional[int] = Field(None, description="Maximum requests per minute")
    cache_type: str = Field('disk', description="Cache type ('disk', 'memory', 'none')")
    stop_on_task_error: bool = Field(False, description="Whether to stop on task error")


class BaseCrew:
    """
    Base crew class with common functionality for all crew types.
    Provides standardized initialization and task management.
    """

    def __init__(
            self,
            config: CrewConfig,
            agents: List[BaseAgent] = None,
            tasks: List[Task] = None,
    ):
        """
        Initialize a base crew with common attributes.

        Args:
            config: Crew configuration
            agents: List of agents in the crew
            tasks: List of tasks for the crew
        """
        self.config = config
        self.agents = agents or []
        self.tasks = tasks or []

        # Initialize shared memory if not provided
        if config.memory is None:
            self.config.memory = SharedMemory(crew_name=config.name)
        else:
            self.config.memory = config.memory

        # Task execution history
        self.task_history = []

        # CrewAI crew instance (initialized lazily)
        self._crew = None

        logger.info(f"Initialized {self.__class__.__name__}: {config.name}")

    def add_agent(self, agent: BaseAgent) -> None:
        """
        Add an agent to the crew.

        Args:
            agent: Agent to add
        """
        self.agents.append(agent)
        # Reset crew to rebuild with new agent
        self._crew = None
        logger.info(f"Added agent {agent.name} to crew {self.config.name}")

    def add_agents(self, agents: List[BaseAgent]) -> None:
        """
        Add multiple agents to the crew.

        Args:
            agents: List of agents to add
        """
        self.agents.extend(agents)
        # Reset crew to rebuild with new agents
        self._crew = None
        logger.info(f"Added {len(agents)} agents to crew {self.config.name}")

    def add_task(self, task: Task) -> None:
        """
        Add a task to the crew.

        Args:
            task: Task to add
        """
        self.tasks.append(task)
        # Reset crew to rebuild with new task
        self._crew = None
        logger.info(f"Added task to crew {self.config.name}")

    def add_tasks(self, tasks: List[Task]) -> None:
        """
        Add multiple tasks to the crew.

        Args:
            tasks: List of tasks to add
        """
        self.tasks.extend(tasks)
        # Reset crew to rebuild with new tasks
        self._crew = None
        logger.info(f"Added {len(tasks)} tasks to crew {self.config.name}")

    def create_task(
            self,
            description: str,
            agent: Optional[BaseAgent] = None,
            expected_output: Optional[str] = None,
            context: Optional[str] = None,
            async_execution: bool = False,
            output_file: Optional[str] = None,
            callback: Optional[Callable] = None
    ) -> Task:
        """
        Create a new task and add it to the crew.

        Args:
            description: Task description
            agent: Agent to assign the task to (if None, will be assigned later)
            expected_output: Expected output format
            context: Additional context for the task
            async_execution: Whether to execute the task asynchronously
            output_file: File to write the output to
            callback: Callback function to execute after task completion

        Returns:
            Task: Created task instance
        """
        agent_instance = None
        if agent is not None:
            agent_instance = agent.get_crew_agent()

        task = Task(
            description=description,
            agent=agent_instance,
            expected_output=expected_output,
            context=context,
            async_execution=async_execution,
            output_file=output_file,
            callback=callback
        )

        self.add_task(task)
        return task

    def get_crew(self) -> Crew:
        """
        Get or create the CrewAI crew instance.

        Returns:
            Crew: CrewAI crew instance
        """
        if self._crew is None:
            # Convert BaseAgent instances to CrewAI Agent instances
            crew_agents = [agent.get_crew_agent() for agent in self.agents]

            # Create the crew
            self._crew = Crew(
                agents=crew_agents,
                tasks=self.tasks,
                verbose=self.config.verbose,
                sequential=self.config.sequential,
                memory=self.config.memory,
                max_rpm=self.config.max_rpm,
                cache=self.config.cache_type != 'none',
                cache_type=self.config.cache_type if self.config.cache_type != 'none' else None,
                process_inputs=self._process_task_inputs if hasattr(self, '_process_task_inputs') else None,
                manager_llm_config=self._get_manager_llm_config() if hasattr(self, '_get_manager_llm_config') else None
            )

        return self._crew

    async def run(self) -> str:
        """
        Run the crew's tasks.

        Returns:
            str: Execution result
        """
        crew = self.get_crew()

        try:
            # Record start time
            start_time = datetime.datetime.now()

            # Execute crew tasks
            result = await crew.run_async()

            # Record end time
            end_time = datetime.datetime.now()
            execution_time = (end_time - start_time).total_seconds()

            # Save execution in history
            self.task_history.append({
                "timestamp": start_time.isoformat(),
                "execution_time": execution_time,
                "tasks": [t.description for t in self.tasks],
                "agents": [a.name for a in self.agents],
                "result": result
            })

            logger.info(f"Crew {self.config.name} completed execution in {execution_time:.2f} seconds")
            return result

        except Exception as e:
            logger.error(f"Error executing crew {self.config.name}: {str(e)}")
            if not self.config.stop_on_task_error:
                return f"Error: {str(e)}"
            raise

    def get_task_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get the crew's task execution history.

        Args:
            limit: Maximum number of history items to return

        Returns:
            List[Dict[str, Any]]: Task history
        """
        if limit is not None:
            return self.task_history[-limit:]
        return self.task_history

    def clear_task_history(self) -> None:
        """
        Clear the crew's task execution history.
        """
        self.task_history = []
        logger.info(f"Crew {self.config.name} task history cleared")

    def reset(self) -> None:
        """
        Reset the crew, clearing tasks and history.
        """
        self.tasks = []
        self.clear_task_history()
        self._crew = None
        logger.info(f"Crew {self.config.name} reset")

    async def generate_execution_summary(self) -> Dict[str, Any]:
        """
        Generate a summary of the crew's execution.

        Returns:
            Dict[str, Any]: Execution summary
        """
        if not self.task_history:
            return {"error": "No execution history available"}

        last_execution = self.task_history[-1]

        # Calculate statistics
        total_time = last_execution["execution_time"]
        tasks_count = len(self.tasks)
        agents_count = len(self.agents)

        return {
            "crew_name": self.config.name,
            "execution_time": total_time,
            "tasks_completed": tasks_count,
            "agents_involved": agents_count,
            "result_summary": last_execution["result"][:500] + "..." if len(last_execution["result"]) > 500 else
            last_execution["result"],
            "timestamp": last_execution["timestamp"]
        }


# Factory function to create a base crew with default configuration
def create_base_crew(
        name: str,
        description: str,
        agents: List[BaseAgent] = None,
        tasks: List[Task] = None,
        **kwargs
) -> BaseCrew:
    """
    Create a BaseCrew with default configuration.

    Args:
        name: Crew name
        description: Crew description
        agents: List of agents in the crew
        tasks: List of tasks for the crew
        **kwargs: Additional configuration parameters

    Returns:
        BaseCrew: Configured crew instance
    """
    config = CrewConfig(
        name=name,
        description=description,
        **kwargs
    )

    return BaseCrew(
        config=config,
        agents=agents or [],
        tasks=tasks or []
    )