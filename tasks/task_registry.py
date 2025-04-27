"""
Task registry for Snowflake Agent application.
Handles registration, discovery, and loading of different task types.
"""

import logging
import importlib
import inspect
from typing import Dict, List, Any, Optional, Type, Callable, Union

from crewai import Task
from pydantic import BaseModel, Field

# Configure logger
logger = logging.getLogger(__name__)


class TaskMetadata(BaseModel):
    """Metadata for a registered task type."""

    name: str = Field(..., description="Unique name for the task type")
    description: str = Field(..., description="Description of what the task does")
    module_path: str = Field(..., description="Import path for the task implementation")
    class_name: str = Field(..., description="Class name for the task implementation")
    parameters: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Required parameters with descriptions and defaults"
    )
    category: str = Field("general", description="Category for grouping related tasks")
    tags: List[str] = Field(default_factory=list, description="Tags for searching/filtering tasks")
    version: str = Field("1.0.0", description="Version of the task implementation")
    factory_method: Optional[str] = Field(
        None, description="Optional factory method name for creating task instances"
    )


class TaskRegistry:
    """
    Registry for CrewAI tasks in the Snowflake Agent application.
    Provides mechanisms for registering, discovering, and loading tasks.
    """

    def __init__(self):
        """Initialize the task registry."""
        self._tasks: Dict[str, TaskMetadata] = {}
        self._task_classes: Dict[str, Type] = {}
        self._task_factories: Dict[str, Callable] = {}

        # Auto-discover built-in tasks
        self._discover_built_in_tasks()

        logger.info(f"TaskRegistry initialized with {len(self._tasks)} tasks")

    def _discover_built_in_tasks(self) -> None:
        """
        Automatically discover and register built-in tasks.
        """
        try:
            # Import the analysis_tasks module
            from tasks import analysis_tasks
            # Find all task-related classes in the module
            self._register_from_module(analysis_tasks, category="analysis")

            # Import the research_tasks module
            from tasks import research_tasks
            # Find all task-related classes in the module
            self._register_from_module(research_tasks, category="research")

            logger.info("Built-in tasks discovered and registered")
        except ImportError as e:
            logger.warning(f"Could not import task modules: {str(e)}")
        except Exception as e:
            logger.error(f"Error during built-in task discovery: {str(e)}")

    def _register_from_module(self, module: Any, category: str = "general") -> None:
        """
        Find and register all task-related classes in a module.

        Args:
            module: Module containing task classes
            category: Category to assign to discovered tasks
        """
        for name, obj in inspect.getmembers(module):
            # Look for classes that might be task implementations
            if inspect.isclass(obj) and name.endswith('Task'):
                # Skip abstract base classes
                if inspect.isabstract(obj):
                    continue

                # Extract metadata from the class
                description = obj.__doc__.split('\n')[0].strip() if obj.__doc__ else "No description available"

                # Create task registration
                task_name = name.replace('Task', '').lower()
                self.register_task(
                    name=task_name,
                    description=description,
                    module_path=module.__name__,
                    class_name=name,
                    category=category
                )

    def register_task(
            self,
            name: str,
            description: str,
            module_path: str,
            class_name: str,
            parameters: Dict[str, Dict[str, Any]] = None,
            category: str = "general",
            tags: List[str] = None,
            version: str = "1.0.0",
            factory_method: str = None
    ) -> None:
        """
        Register a task type with the registry.

        Args:
            name: Unique name for the task
            description: Description of what the task does
            module_path: Import path for the task implementation
            class_name: Class name for the task implementation
            parameters: Required parameters with descriptions and defaults
            category: Category for grouping related tasks
            tags: Tags for searching/filtering tasks
            version: Version of the task implementation
            factory_method: Optional factory method name for creating instances
        """
        # Create task metadata
        metadata = TaskMetadata(
            name=name,
            description=description,
            module_path=module_path,
            class_name=class_name,
            parameters=parameters or {},
            category=category,
            tags=tags or [],
            version=version,
            factory_method=factory_method
        )

        # Check for duplicate task names
        if name in self._tasks:
            logger.warning(f"Overwriting existing task registration: {name}")

        # Register the task
        self._tasks[name] = metadata
        logger.info(f"Registered task: {name} ({module_path}.{class_name})")

        # Clear cached class and factory references
        if name in self._task_classes:
            del self._task_classes[name]
        if name in self._task_factories:
            del self._task_factories[name]

    def get_task_metadata(self, name: str) -> Optional[TaskMetadata]:
        """
        Get metadata for a registered task.

        Args:
            name: Name of the task

        Returns:
            Optional[TaskMetadata]: Task metadata or None if not found
        """
        return self._tasks.get(name)

    def list_tasks(self, category: str = None, tags: List[str] = None) -> List[TaskMetadata]:
        """
        List registered tasks, optionally filtered by category or tags.

        Args:
            category: Optional category filter
            tags: Optional tags filter (tasks must have all specified tags)

        Returns:
            List[TaskMetadata]: List of matching task metadata
        """
        tasks = list(self._tasks.values())

        # Apply category filter
        if category:
            tasks = [t for t in tasks if t.category == category]

        # Apply tags filter
        if tags:
            tasks = [t for t in tasks if all(tag in t.tags for tag in tags)]

        return tasks

    def get_categories(self) -> List[str]:
        """
        Get all registered task categories.

        Returns:
            List[str]: List of unique task categories
        """
        return sorted(set(t.category for t in self._tasks.values()))

    def get_tags(self) -> List[str]:
        """
        Get all registered task tags.

        Returns:
            List[str]: List of unique task tags
        """
        all_tags = set()
        for task in self._tasks.values():
            all_tags.update(task.tags)
        return sorted(all_tags)

    def _load_task_class(self, name: str) -> Type:
        """
        Load the class for a registered task.

        Args:
            name: Name of the task

        Returns:
            Type: Task class

        Raises:
            ValueError: If task is not registered or class cannot be loaded
        """
        # Check if already loaded
        if name in self._task_classes:
            return self._task_classes[name]

        # Get task metadata
        metadata = self.get_task_metadata(name)
        if not metadata:
            raise ValueError(f"Task not registered: {name}")

        try:
            # Import the module
            module = importlib.import_module(metadata.module_path)

            # Get the class
            task_class = getattr(module, metadata.class_name)

            # Cache and return the class
            self._task_classes[name] = task_class
            return task_class
        except ImportError:
            raise ValueError(f"Could not import module for task: {metadata.module_path}")
        except AttributeError:
            raise ValueError(f"Could not find class for task: {metadata.class_name}")

    def _get_task_factory(self, name: str) -> Callable:
        """
        Get the factory function for creating task instances.

        Args:
            name: Name of the task

        Returns:
            Callable: Factory function

        Raises:
            ValueError: If task is not registered or factory cannot be loaded
        """
        # Check if already loaded
        if name in self._task_factories:
            return self._task_factories[name]

        # Get task metadata
        metadata = self.get_task_metadata(name)
        if not metadata:
            raise ValueError(f"Task not registered: {name}")

        # Get task class
        task_class = self._load_task_class(name)

        # Get factory method if specified
        if metadata.factory_method:
            try:
                factory = getattr(task_class, metadata.factory_method)
                self._task_factories[name] = factory
                return factory
            except AttributeError:
                raise ValueError(f"Could not find factory method: {metadata.factory_method}")

        # Default factory function: class constructor
        def default_factory(**kwargs):
            return task_class(**kwargs)

        self._task_factories[name] = default_factory
        return default_factory

    def create_task(
            self,
            name: str,
            description: str = None,
            agent=None,
            expected_output: str = None,
            context: str = None,
            async_execution: bool = False,
            **kwargs
    ) -> Task:
        """
        Create a task instance based on a registered task type.

        Args:
            name: Name of the registered task
            description: Task description (overrides default)
            agent: Agent to assign the task to
            expected_output: Expected output format
            context: Additional context for the task
            async_execution: Whether to execute the task asynchronously
            **kwargs: Additional task parameters

        Returns:
            Task: Instantiated task

        Raises:
            ValueError: If task cannot be created
        """
        # Get task metadata
        metadata = self.get_task_metadata(name)
        if not metadata:
            raise ValueError(f"Task not registered: {name}")

        try:
            # Get task factory
            factory = self._get_task_factory(name)

            # Merge default parameters from metadata with provided kwargs
            merged_params = {}
            for param_name, param_info in metadata.parameters.items():
                if 'default' in param_info and param_name not in kwargs:
                    merged_params[param_name] = param_info['default']

            # Override with user-provided parameters
            merged_params.update(kwargs)

            # Create base CrewAI task with task-specific details
            task_instance = factory(**merged_params)

            # If the task factory returns a CrewAI Task, use it directly
            if isinstance(task_instance, Task):
                # Update with provided overrides
                if description is not None:
                    task_instance.description = description
                if agent is not None:
                    task_instance.agent = agent
                if expected_output is not None:
                    task_instance.expected_output = expected_output
                if context is not None:
                    task_instance.context = context
                task_instance.async_execution = async_execution
                return task_instance

            # Otherwise, create a CrewAI Task using the returned object as input
            # This allows task factories to return data that gets wrapped in a Task
            task_description = description or f"{metadata.name}: {metadata.description}"
            return Task(
                description=task_description,
                agent=agent,
                expected_output=expected_output,
                context=context,
                async_execution=async_execution,
                **merged_params
            )
        except Exception as e:
            logger.error(f"Error creating task {name}: {str(e)}")
            raise ValueError(f"Could not create task {name}: {str(e)}")

    def unregister_task(self, name: str) -> bool:
        """
        Unregister a task from the registry.

        Args:
            name: Name of the task to unregister

        Returns:
            bool: True if task was unregistered, False if not found
        """
        if name in self._tasks:
            del self._tasks[name]

            # Clean up cached references
            if name in self._task_classes:
                del self._task_classes[name]
            if name in self._task_factories:
                del self._task_factories[name]

            logger.info(f"Unregistered task: {name}")
            return True

        return False


# Create singleton instance
task_registry = TaskRegistry()


# Helper functions for direct usage
def register_task(
        name: str,
        description: str,
        module_path: str,
        class_name: str,
        **kwargs
) -> None:
    """
    Register a task with the global registry.

    Args:
        name: Task name
        description: Task description
        module_path: Import path
        class_name: Class name
        **kwargs: Additional registration parameters
    """
    task_registry.register_task(
        name=name,
        description=description,
        module_path=module_path,
        class_name=class_name,
        **kwargs
    )


def create_task(name: str, **kwargs) -> Task:
    """
    Create a task from the global registry.

    Args:
        name: Task name
        **kwargs: Task parameters

    Returns:
        Task: Instantiated task
    """
    return task_registry.create_task(name, **kwargs)


def list_tasks(category: str = None, tags: List[str] = None) -> List[TaskMetadata]:
    """
    List tasks from the global registry.

    Args:
        category: Optional category filter
        tags: Optional tags filter

    Returns:
        List[TaskMetadata]: List of task metadata
    """
    return task_registry.list_tasks(category, tags)


# Export classes and helper functions
__all__ = [
    'TaskRegistry',
    'TaskMetadata',
    'task_registry',
    'register_task',
    'create_task',
    'list_tasks'
]