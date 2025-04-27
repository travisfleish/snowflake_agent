from typing import Dict, Any, Optional, List, Set, Tuple
import time
import json
import threading
import logging
from datetime import datetime, timedelta


class SharedMemory:
    """
    In-memory storage for temporary session data in CrewAI workflows.
    Provides thread-safe access to shared data between agents.
    """

    def __init__(self, crew_name: str, ttl: Optional[int] = None):
        """
        Initialize shared memory storage.

        Args:
            crew_name: Name of the crew for namespace isolation
            ttl: Default time-to-live for items in seconds (None for no expiration)
        """
        self.crew_name = crew_name
        self.default_ttl = ttl

        # Main storage dict with namespace isolation
        self._data: Dict[str, Dict[str, Any]] = {}

        # Expiration tracking
        self._expiration: Dict[str, float] = {}

        # Thread lock for concurrent access
        self._lock = threading.RLock()

        # Initialize logger
        self.logger = logging.getLogger(f"shared_memory.{crew_name}")

        # Storage metrics
        self.metrics = {
            "gets": 0,
            "sets": 0,
            "hits": 0,
            "misses": 0,
            "expirations": 0,
            "removals": 0
        }

    def add(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """
        Add an item to the shared memory.

        Args:
            key: Key to store the value under
            value: Value to store
            ttl: Time-to-live in seconds (uses default if None)
        """
        with self._lock:
            self._data[key] = value
            self.metrics["sets"] += 1

            # Set expiration if ttl is provided
            if ttl is not None or self.default_ttl is not None:
                expiration_time = time.time() + (ttl if ttl is not None else self.default_ttl)
                self._expiration[key] = expiration_time

            self.logger.debug(f"Added key: {key}")

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get an item from shared memory.

        Args:
            key: Key to retrieve
            default: Default value if key not found

        Returns:
            Value associated with key or default if not found
        """
        with self._lock:
            self.metrics["gets"] += 1

            # Check if key exists and not expired
            if key in self._data:
                if key in self._expiration and time.time() > self._expiration[key]:
                    # Key has expired
                    self.remove(key)
                    self.metrics["expirations"] += 1
                    self.metrics["misses"] += 1
                    return default

                self.metrics["hits"] += 1
                return self._data[key]

            self.metrics["misses"] += 1
            return default

    def remove(self, key: str) -> bool:
        """
        Remove an item from shared memory.

        Args:
            key: Key to remove

        Returns:
            True if key was found and removed, False otherwise
        """
        with self._lock:
            if key in self._data:
                del self._data[key]
                if key in self._expiration:
                    del self._expiration[key]

                self.metrics["removals"] += 1
                self.logger.debug(f"Removed key: {key}")
                return True

            return False

    def contains(self, key: str) -> bool:
        """
        Check if key exists in shared memory and is not expired.

        Args:
            key: Key to check

        Returns:
            True if key exists and is not expired, False otherwise
        """
        with self._lock:
            if key not in self._data:
                return False

            if key in self._expiration and time.time() > self._expiration[key]:
                # Key has expired - clean it up
                self.remove(key)
                return False

            return True

    def update(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """
        Update an existing item in shared memory.

        Args:
            key: Key to update
            value: New value
            ttl: New time-to-live in seconds (keeps existing if None)

        Returns:
            True if key existed and was updated, False otherwise
        """
        with self._lock:
            if not self.contains(key):
                return False

            self._data[key] = value

            # Update expiration if ttl is provided
            if ttl is not None:
                expiration_time = time.time() + ttl
                self._expiration[key] = expiration_time

            self.logger.debug(f"Updated key: {key}")
            return True

    def clear(self) -> None:
        """Clear all data in shared memory."""
        with self._lock:
            self._data.clear()
            self._expiration.clear()
            self.logger.info("Memory cleared")

    def get_all(self) -> Dict[str, Any]:
        """
        Get all non-expired items in shared memory.

        Returns:
            Dict of all keys and values that haven't expired
        """
        with self._lock:
            # Clean expired entries
            self._clean_expired()

            # Return a copy of the data to prevent external modification
            return dict(self._data)

    def get_keys(self) -> List[str]:
        """
        Get all keys in shared memory.

        Returns:
            List of all non-expired keys
        """
        with self._lock:
            # Clean expired entries
            self._clean_expired()

            return list(self._data.keys())

    def _clean_expired(self) -> int:
        """
        Remove all expired items from shared memory.

        Returns:
            Number of items removed
        """
        now = time.time()
        expired_keys = [
            key for key, expiry in self._expiration.items()
            if now > expiry
        ]

        for key in expired_keys:
            self.remove(key)
            self.metrics["expirations"] += 1

        return len(expired_keys)

    def set_namespace(self, namespace: str, data: Dict[str, Any], ttl: Optional[int] = None) -> None:
        """
        Store a dictionary of values under a namespace.

        Args:
            namespace: Namespace to store under
            data: Dictionary of data to store
            ttl: Time-to-live in seconds for all items
        """
        with self._lock:
            for key, value in data.items():
                namespaced_key = f"{namespace}:{key}"
                self.add(namespaced_key, value, ttl)

    def get_namespace(self, namespace: str) -> Dict[str, Any]:
        """
        Get all keys and values in a namespace.

        Args:
            namespace: Namespace to retrieve

        Returns:
            Dictionary of keys and values in the namespace
        """
        with self._lock:
            # Clean expired entries first
            self._clean_expired()

            prefix = f"{namespace}:"
            result = {}

            for key in self._data:
                if key.startswith(prefix):
                    # Extract the original key name without namespace
                    original_key = key[len(prefix):]
                    result[original_key] = self._data[key]

            return result

    def remove_namespace(self, namespace: str) -> int:
        """
        Remove all keys in a namespace.

        Args:
            namespace: Namespace to remove

        Returns:
            Number of keys removed
        """
        with self._lock:
            prefix = f"{namespace}:"
            to_remove = [key for key in self._data if key.startswith(prefix)]

            for key in to_remove:
                self.remove(key)

            return len(to_remove)

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get usage metrics.

        Returns:
            Dictionary of storage metrics
        """
        with self._lock:
            # Add current item count to metrics
            current_metrics = dict(self.metrics)
            current_metrics["current_items"] = len(self._data)
            current_metrics["hit_ratio"] = (
                current_metrics["hits"] / current_metrics["gets"]
                if current_metrics["gets"] > 0 else 0
            )

            return current_metrics

    def to_json(self) -> str:
        """
        Serialize storage to JSON.

        Returns:
            JSON string representation of storage
        """
        with self._lock:
            # Clean expired entries
            self._clean_expired()

            # Prepare serializable data
            serializable_data = {}
            for key, value in self._data.items():
                try:
                    # Test if value is JSON serializable
                    json.dumps({key: value})
                    serializable_data[key] = value
                except (TypeError, OverflowError):
                    # Skip values that can't be serialized
                    self.logger.warning(f"Skipping non-serializable value for key: {key}")

            return json.dumps(serializable_data)

    def from_json(self, json_str: str) -> None:
        """
        Load storage from JSON.

        Args:
            json_str: JSON string to load
        """
        with self._lock:
            try:
                loaded_data = json.loads(json_str)
                if isinstance(loaded_data, dict):
                    # Reset current data
                    self._data.clear()
                    self._expiration.clear()

                    # Load new data
                    for key, value in loaded_data.items():
                        self._data[key] = value

                    self.logger.info(f"Loaded {len(loaded_data)} items from JSON")
                else:
                    self.logger.error("Invalid JSON format: root must be an object")
            except json.JSONDecodeError:
                self.logger.error("Failed to parse JSON data")