"""
Base agent implementation for Snowflake Agent system.
Provides foundational functionality for all agent types.
"""

import logging
from typing import List, Dict, Any, Optional, Union, Callable

from crewai import Agent, Task, Crew
from pydantic import BaseModel, Field

from utils.llm_cache import CachedLLMClient
from config.settings import settings
from utils.prompt_templates import BasePromptTemplates

# Configure logger
logger = logging.getLogger(__name__)


class BaseAgent:
    """
    Base agent class with common functionality for all agent types.
    Provides standardized initialization and CrewAI integration.
    """

    def __init__(
        self,
        name: str,
        role: str,
        goal: str,
        backstory: str = None,
        verbose: bool = False,
        allow_delegation: bool = True,
        tools: List[Any] = None,
        memory: Any = None,
        llm_client: CachedLLMClient = None,
        llm_config: Dict[str, Any] = None,
        provider: str = "openai",
        **kwargs
    ):
        """
        Initialize a base agent with common attributes.

        Args:
            name: Agent's name
            role: Agent's role description
            goal: Agent's main objective
            backstory: Agent's background story (optional)
            verbose: Whether to enable verbose output
            allow_delegation: Whether agent can delegate tasks
            tools: List of tools the agent can use
            memory: Agent memory instance
            llm_client: Custom LLM client instance
            llm_config: LLM configuration parameters
            provider: LLM provider name (e.g., "openai", "anthropic")
            **kwargs: Additional agent-specific parameters
        """
        self.name = name
        self.role = role
        self.goal = goal
        self.backstory = backstory or self._generate_default_backstory()
        self.verbose = verbose
        self.allow_delegation = allow_delegation
        self.tools = tools or []
        self.memory = memory
        self.provider = provider
        self.task_history = []

        # Initialize LLM config with defaults from settings
        self.llm_config = {
            "model": settings.openai.model_name,
            "temperature": settings.openai.temperature,
            "max_tokens": settings.openai.max_tokens
        }

        # Override with any provided config
        if llm_config:
            self.llm_config.update(llm_config)

        # Set up LLM client
        self._llm_client = llm_client or self._initialize_llm_client()

        # Additional attributes from kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)

        # CrewAI agent instance (initialized lazily)
        self._crew_agent = None

        logger.info(f"Initialized {self.__class__.__name__}: {self.name} ({self.role})")

    def _initialize_llm_client(self) -> CachedLLMClient:
        """
        Initialize the default LLM client based on provider.

        Returns:
            CachedLLMClient: Configured LLM client
        """
        if self.provider.lower() == "openai":
            return CachedLLMClient(
                api_key=settings.openai.api_key,
                default_model=settings.openai.model_name,
                organization=settings.openai.organization
            )
        elif self.provider.lower() == "anthropic":
            # This would need to be implemented based on your anthropic client
            raise NotImplementedError("Anthropic provider not yet implemented")
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def _generate_default_backstory(self) -> str:
        """
        Generate a default backstory based on role and goal.

        Returns:
            str: Default backstory
        """
        return (
            f"As a {self.role}, my purpose is to {self.goal}. "
            "I have extensive experience in data analysis and problem-solving. "
            "I am detail-oriented, efficient, and focused on delivering high-quality results."
        )

    def add_tool(self, tool: Any) -> None:
        """
        Add a new tool to the agent's toolkit.

        Args:
            tool: Tool to add
        """
        self.tools.append(tool)
        # Reset crew agent to rebuild with new tools
        self._crew_agent = None

    def add_tools(self, tools: List[Any]) -> None:
        """
        Add multiple tools to the agent's toolkit.

        Args:
            tools: List of tools to add
        """
        self.tools.extend(tools)
        # Reset crew agent to rebuild with new tools
        self._crew_agent = None

    def get_crew_agent(self) -> Agent:
        """
        Get or create the CrewAI agent instance.

        Returns:
            Agent: CrewAI agent instance
        """
        if self._crew_agent is None:
            self._crew_agent = Agent(
                name=self.name,
                role=self.role,
                goal=self.goal,
                backstory=self.backstory,
                verbose=self.verbose,
                allow_delegation=self.allow_delegation,
                tools=self.tools,
                llm_config=self.llm_config
            )
        return self._crew_agent

    async def execute_task(self, task: Union[Task, str], **kwargs) -> str:
        """
        Execute a task with this agent.

        Args:
            task: Task instance or task description
            **kwargs: Additional task parameters

        Returns:
            str: Task execution result
        """
        crew_agent = self.get_crew_agent()

        # If task is a string, create a Task object
        if isinstance(task, str):
            from crewai import Task
            task = Task(
                description=task,
                agent=crew_agent,
                **kwargs
            )

        # Create a single-agent crew
        from crewai import Crew
        crew = Crew(
            agents=[crew_agent],
            tasks=[task],
            verbose=self.verbose
        )

        # Execute the task
        result = await crew.run_async()

        # Store in task history
        self.task_history.append({
            "task": task,
            "result": result,
            "timestamp": datetime.datetime.now().isoformat()
        })

        return result

    def thinking(self, thought: str) -> None:
        """
        Log the agent's thinking process.

        Args:
            thought: Agent's thought to log
        """
        if self.verbose:
            logger.info(f"[{self.name} thinking]: {thought}")

    def __repr__(self) -> str:
        """
        String representation of the agent.

        Returns:
            str: Agent description
        """
        return f"{self.__class__.__name__}(name='{self.name}', role='{self.role}')"

    # Enhanced Memory Integration
    def remember(self, key: str, value: Any) -> None:
        """
        Store information in agent's memory.

        Args:
            key: Memory key
            value: Value to store
        """
        if self.memory:
            self.memory.add(key, value)
            if self.verbose:
                logger.debug(f"[{self.name}] Remembered: {key}")

    def recall(self, key: str, default: Any = None) -> Any:
        """
        Retrieve information from agent's memory.

        Args:
            key: Memory key
            default: Default value if key not found

        Returns:
            Any: Retrieved value or default
        """
        if self.memory:
            value = self.memory.get(key)
            if self.verbose and value is not None:
                logger.debug(f"[{self.name}] Recalled: {key}")
            return value if value is not None else default
        return default

    def forget(self, key: str) -> bool:
        """
        Remove information from agent's memory.

        Args:
            key: Memory key to forget

        Returns:
            bool: True if successfully forgotten, False otherwise
        """
        if self.memory and key in self.memory:
            self.memory.remove(key)
            if self.verbose:
                logger.debug(f"[{self.name}] Forgot: {key}")
            return True
        return False

    # Task History Management
    def get_task_history(self, limit: int = None) -> List[Dict[str, Any]]:
        """
        Get the agent's task execution history.

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
        Clear the agent's task execution history.
        """
        self.task_history = []
        logger.info(f"[{self.name}] Task history cleared")

    # Agent Self-Evaluation
    async def evaluate_performance(self, task_result: str = None, task_idx: int = -1) -> Dict[str, Any]:
        """
        Have the agent evaluate its own performance on a task.

        Args:
            task_result: Task result to evaluate (if None, uses the last task)
            task_idx: Index of the task in history to evaluate

        Returns:
            Dict[str, Any]: Evaluation results
        """
        # Get the task result to evaluate
        if task_result is None:
            if not self.task_history:
                return {"error": "No task history available to evaluate"}
            task_result = self.task_history[task_idx]["result"]
            task = self.task_history[task_idx]["task"]
        else:
            task = "Provided task result"

        prompt = f"""
        Based on my goal to {self.goal}, evaluate my performance on this task:
        
        Task description: {task.description if hasattr(task, 'description') else task}
        
        Task result:
        {task_result}
        
        Provide a score (1-10) and explanation for:
        1. Task completion: Did I fully complete what was asked?
        2. Accuracy: Was my response correct and precise?
        3. Efficiency: Did I solve it in an optimal way?
        4. Usefulness: How valuable is my response for the user?
        
        Finally, suggest one specific improvement I could make next time.
        """

        response = await self._llm_client.generate(prompt)
        eval_content = response['choices'][0]['message']['content']

        return {
            "evaluation": eval_content,
            "task": task,
            "raw_task_result": task_result,
            "timestamp": datetime.datetime.now().isoformat()
        }

    # Dynamic Goal Management
    def update_goal(self, new_goal: str) -> None:
        """
        Update the agent's goal and reset its CrewAI representation.

        Args:
            new_goal: New goal for the agent
        """
        logger.info(f"[{self.name}] Updating goal from '{self.goal}' to '{new_goal}'")
        self.goal = new_goal
        self._crew_agent = None  # Force recreation

    def update_role(self, new_role: str) -> None:
        """
        Update the agent's role and reset its CrewAI representation.

        Args:
            new_role: New role for the agent
        """
        logger.info(f"[{self.name}] Updating role from '{self.role}' to '{new_role}'")
        self.role = new_role
        self._crew_agent = None  # Force recreation

    def update_backstory(self, new_backstory: str) -> None:
        """
        Update the agent's backstory and reset its CrewAI representation.

        Args:
            new_backstory: New backstory for the agent
        """
        logger.info(f"[{self.name}] Updating backstory")
        self.backstory = new_backstory
        self._crew_agent = None  # Force recreation