"""
LLM Cache System for Snowflake Agent.
Provides caching mechanisms to reduce API costs by storing LLM outputs.
"""

import os
import json
import time
import hashlib
import logging
import threading
from typing import Dict, Any, Optional, Union, Tuple
from pathlib import Path
import pickle
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class LLMCache:
    """
    Caching system for LLM outputs to reduce API costs.
    Supports both in-memory and disk-based caching with configurable expiration.
    """

    def __init__(self,
                 cache_dir: str = None,
                 memory_cache_size: int = 1000,
                 disk_cache_enabled: bool = True,
                 default_ttl: int = 86400,  # Default TTL: 1 day in seconds
                 compression_enabled: bool = True):
        """
        Initialize the LLM caching system.

        Args:
            cache_dir: Directory for disk cache (defaults to ./cache)
            memory_cache_size: Maximum number of items in memory cache
            disk_cache_enabled: Whether to use disk-based caching
            default_ttl: Default time-to-live for cache entries (seconds)
            compression_enabled: Whether to compress disk cache entries
        """
        self.memory_cache: Dict[str, Dict[str, Any]] = {}
        self.memory_cache_size = memory_cache_size
        self.default_ttl = default_ttl
        self.disk_cache_enabled = disk_cache_enabled
        self.compression_enabled = compression_enabled

        # Set up cache directory
        if cache_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            cache_dir = os.path.join(base_dir, "cache")

        self.cache_dir = cache_dir

        if self.disk_cache_enabled:
            os.makedirs(self.cache_dir, exist_ok=True)

        # Add thread lock for thread safety
        self._lock = threading.RLock()

        # Load stats
        self.stats = {
            "memory_hits": 0,
            "disk_hits": 0,
            "misses": 0,
            "total_requests": 0,
        }

        logger.info(f"LLM Cache initialized (memory_size={memory_cache_size}, "
                    f"disk_enabled={disk_cache_enabled}, ttl={default_ttl}s)")

    def _generate_key(self, prompt: str, model: str, params: Dict[str, Any]) -> str:
        """
        Generate a unique cache key for the input.

        Args:
            prompt: The prompt text
            model: The model identifier
            params: Additional parameters affecting the output

        Returns:
            str: Unique hash key
        """
        # Create a normalized representation of the inputs
        key_parts = [
            prompt.strip(),
            model,
            # Sort parameters to ensure consistent ordering
            json.dumps(params, sort_keys=True)
        ]

        # Create a hash of the key parts
        key_string = "::".join(key_parts)
        return hashlib.sha256(key_string.encode('utf-8')).hexdigest()

    def _is_cache_valid(self, entry: Dict[str, Any]) -> bool:
        """
        Check if a cache entry is still valid (not expired).

        Args:
            entry: Cache entry to check

        Returns:
            bool: True if valid, False if expired
        """
        expiration = entry.get('expiration')
        if expiration is None:
            return True

        return time.time() < expiration

    def _save_to_disk(self, key: str, entry: Dict[str, Any]) -> bool:
        """
        Save a cache entry to disk.

        Args:
            key: Cache key
            entry: Cache entry data

        Returns:
            bool: True if successful, False otherwise
        """
        if not self.disk_cache_enabled:
            return False

        try:
            # Create file path based on key
            path = os.path.join(self.cache_dir, f"{key}.cache")

            # Save entry to disk
            with open(path, 'wb') as f:
                pickle.dump(entry, f, protocol=pickle.HIGHEST_PROTOCOL)

            return True
        except Exception as e:
            logger.error(f"Error saving cache to disk: {str(e)}")
            return False

    def _load_from_disk(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Load a cache entry from disk.

        Args:
            key: Cache key

        Returns:
            Optional[Dict[str, Any]]: Cache entry if found and valid, None otherwise
        """
        if not self.disk_cache_enabled:
            return None

        try:
            # Create file path based on key
            path = os.path.join(self.cache_dir, f"{key}.cache")

            # Check if file exists
            if not os.path.exists(path):
                return None

            # Load entry from disk
            with open(path, 'rb') as f:
                entry = pickle.load(f)

            # Check if entry is valid
            if not self._is_cache_valid(entry):
                # Remove expired entry
                os.remove(path)
                return None

            return entry
        except Exception as e:
            logger.error(f"Error loading cache from disk: {str(e)}")
            return None

    def get(self, prompt: str, model: str, params: Dict[str, Any]) -> Tuple[bool, Optional[Any]]:
        """
        Get a cached LLM output if available.

        Args:
            prompt: The prompt text
            model: The model identifier
            params: Additional parameters affecting the output

        Returns:
            Tuple[bool, Optional[Any]]: (found, value) tuple
        """
        with self._lock:
            self.stats["total_requests"] += 1

            # Generate cache key
            key = self._generate_key(prompt, model, params)

            # Check memory cache first
            if key in self.memory_cache:
                entry = self.memory_cache[key]

                # Check if entry is valid
                if self._is_cache_valid(entry):
                    self.stats["memory_hits"] += 1
                    logger.debug(f"Memory cache hit for key: {key[:8]}...")
                    return True, entry['value']
                else:
                    # Remove expired entry
                    del self.memory_cache[key]

            # If not in memory, check disk cache
            disk_entry = self._load_from_disk(key)
            if disk_entry is not None:
                # Add to memory cache for faster future access
                self.memory_cache[key] = disk_entry

                # Enforce memory cache size limit
                if len(self.memory_cache) > self.memory_cache_size:
                    # Remove oldest entry (simple approach)
                    oldest_key = next(iter(self.memory_cache))
                    del self.memory_cache[oldest_key]

                self.stats["disk_hits"] += 1
                logger.debug(f"Disk cache hit for key: {key[:8]}...")
                return True, disk_entry['value']

            # Cache miss
            self.stats["misses"] += 1
            logger.debug(f"Cache miss for key: {key[:8]}...")
            return False, None

    def set(self, prompt: str, model: str, params: Dict[str, Any],
            value: Any, ttl: Optional[int] = None) -> bool:
        """
        Store an LLM output in the cache.

        Args:
            prompt: The prompt text
            model: The model identifier
            params: Additional parameters affecting the output
            value: The LLM output to cache
            ttl: Time-to-live in seconds (None for default TTL)

        Returns:
            bool: True if successful, False otherwise
        """
        with self._lock:
            # Generate cache key
            key = self._generate_key(prompt, model, params)

            # Create cache entry
            entry = {
                'value': value,
                'created_at': time.time(),
                'expiration': time.time() + (ttl if ttl is not None else self.default_ttl),
                'prompt': prompt,
                'model': model,
                'params': params
            }

            # Add to memory cache
            self.memory_cache[key] = entry

            # Enforce memory cache size limit
            if len(self.memory_cache) > self.memory_cache_size:
                # Remove oldest entry (simple approach)
                oldest_key = next(iter(self.memory_cache))
                del self.memory_cache[oldest_key]

            # Save to disk if enabled
            if self.disk_cache_enabled:
                self._save_to_disk(key, entry)

            logger.debug(f"Cached output for key: {key[:8]}...")
            return True

    def invalidate(self, prompt: str = None, model: str = None,
                   params: Dict[str, Any] = None) -> int:
        """
        Invalidate cache entries matching the provided criteria.

        Args:
            prompt: Optional prompt text filter
            model: Optional model identifier filter
            params: Optional parameters filter

        Returns:
            int: Number of entries invalidated
        """
        with self._lock:
            count = 0

            # If all parameters are None, clear entire cache
            if prompt is None and model is None and params is None:
                count = len(self.memory_cache)
                self.memory_cache.clear()

                # Clear disk cache
                if self.disk_cache_enabled:
                    for file in os.listdir(self.cache_dir):
                        if file.endswith('.cache'):
                            os.remove(os.path.join(self.cache_dir, file))
            else:
                # Selective invalidation
                # For memory cache
                keys_to_remove = []
                for key, entry in self.memory_cache.items():
                    if ((prompt is None or entry['prompt'] == prompt) and
                            (model is None or entry['model'] == model) and
                            (params is None or all(entry['params'].get(k) == v for k, v in params.items()))):
                        keys_to_remove.append(key)

                # Remove matched entries
                for key in keys_to_remove:
                    del self.memory_cache[key]
                    count += 1

                    # Remove from disk if enabled
                    if self.disk_cache_enabled:
                        path = os.path.join(self.cache_dir, f"{key}.cache")
                        if os.path.exists(path):
                            os.remove(path)

            logger.info(f"Invalidated {count} cache entries")
            return count

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dict[str, Any]: Cache statistics
        """
        with self._lock:
            # Calculate hit ratios
            total = self.stats["total_requests"]
            memory_hit_ratio = self.stats["memory_hits"] / total if total > 0 else 0
            disk_hit_ratio = self.stats["disk_hits"] / total if total > 0 else 0
            total_hit_ratio = (self.stats["memory_hits"] + self.stats["disk_hits"]) / total if total > 0 else 0

            # Calculate disk cache size
            disk_size = 0
            if self.disk_cache_enabled:
                for file in os.listdir(self.cache_dir):
                    if file.endswith('.cache'):
                        disk_size += os.path.getsize(os.path.join(self.cache_dir, file))

            return {
                **self.stats,
                "memory_cache_size": len(self.memory_cache),
                "memory_cache_limit": self.memory_cache_size,
                "disk_cache_size_bytes": disk_size,
                "memory_hit_ratio": memory_hit_ratio,
                "disk_hit_ratio": disk_hit_ratio,
                "total_hit_ratio": total_hit_ratio
            }

    def clear_expired(self) -> int:
        """
        Clear all expired cache entries.

        Returns:
            int: Number of entries cleared
        """
        with self._lock:
            count = 0

            # Clear expired memory cache entries
            current_time = time.time()
            keys_to_remove = [
                key for key, entry in self.memory_cache.items()
                if entry.get('expiration') and current_time > entry['expiration']
            ]

            for key in keys_to_remove:
                del self.memory_cache[key]
                count += 1

            # Clear expired disk cache entries
            if self.disk_cache_enabled:
                for file in os.listdir(self.cache_dir):
                    if file.endswith('.cache'):
                        path = os.path.join(self.cache_dir, file)
                        try:
                            with open(path, 'rb') as f:
                                entry = pickle.load(f)

                            if entry.get('expiration') and current_time > entry['expiration']:
                                os.remove(path)
                                count += 1
                        except Exception as e:
                            logger.error(f"Error processing cache file {file}: {str(e)}")
                            # Remove corrupted file
                            os.remove(path)
                            count += 1

            logger.info(f"Cleared {count} expired cache entries")
            return count


# Create a singleton instance
llm_cache = LLMCache()


# Helper functions for direct usage
def cache_llm_response(prompt: str, model: str, params: Dict[str, Any],
                       response: Any, ttl: Optional[int] = None) -> bool:
    """
    Cache an LLM response.

    Args:
        prompt: The input prompt
        model: The model used
        params: Additional parameters
        response: The LLM response to cache
        ttl: Optional time-to-live in seconds

    Returns:
        bool: True if successful, False otherwise
    """
    return llm_cache.set(prompt, model, params, response, ttl)


def get_cached_llm_response(prompt: str, model: str,
                            params: Dict[str, Any]) -> Tuple[bool, Optional[Any]]:
    """
    Get a cached LLM response if available.

    Args:
        prompt: The input prompt
        model: The model to use
        params: Additional parameters

    Returns:
        Tuple[bool, Optional[Any]]: (found, response) tuple
    """
    return llm_cache.get(prompt, model, params)


class CachedLLMClient:
    """
    Client wrapper for OpenAI API with caching support.
    Automatically caches responses to reduce API costs.
    """

    def __init__(self, api_key: str,
                 default_model: str = "gpt-4-turbo",
                 cache_ttl: int = 86400,
                 organization: Optional[str] = None):
        """
        Initialize the cached LLM client.

        Args:
            api_key: OpenAI API key
            default_model: Default model to use
            cache_ttl: Default cache TTL in seconds
            organization: Optional organization ID
        """
        # Note: Import here to avoid circular imports
        import openai

        self.client = openai.OpenAI(api_key=api_key, organization=organization)
        self.default_model = default_model
        self.cache_ttl = cache_ttl

        logger.info(f"CachedLLMClient initialized (default_model={default_model})")

    def generate(self, prompt: str,
                 model: Optional[str] = None,
                 temperature: float = 0.7,
                 max_tokens: int = 1000,
                 use_cache: bool = True,
                 **kwargs) -> Dict[str, Any]:
        """
        Generate a response from the LLM with caching.

        Args:
            prompt: The input prompt
            model: Model to use (defaults to self.default_model)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            use_cache: Whether to use caching
            **kwargs: Additional parameters for the API

        Returns:
            Dict[str, Any]: LLM response
        """
        model = model or self.default_model

        # Build parameters dictionary
        params = {
            'temperature': temperature,
            'max_tokens': max_tokens,
            **kwargs
        }

        # Check cache if enabled
        if use_cache:
            found, cached_response = get_cached_llm_response(prompt, model, params)
            if found:
                return cached_response

        # Generate response from API
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )

            # Convert response to dictionary to ensure it's serializable
            response_dict = {
                'id': response.id,
                'created': response.created,
                'model': response.model,
                'choices': [
                    {
                        'message': {
                            'role': choice.message.role,
                            'content': choice.message.content
                        },
                        'finish_reason': choice.finish_reason
                    }
                    for choice in response.choices
                ],
                'usage': {
                    'prompt_tokens': response.usage.prompt_tokens,
                    'completion_tokens': response.usage.completion_tokens,
                    'total_tokens': response.usage.total_tokens
                }
            }

            # Cache the response if enabled
            if use_cache:
                cache_llm_response(prompt, model, params, response_dict, self.cache_ttl)

            return response_dict

        except Exception as e:
            logger.error(f"Error generating LLM response: {str(e)}")
            raise


# Export classes and helper functions
__all__ = [
    'LLMCache',
    'llm_cache',
    'cache_llm_response',
    'get_cached_llm_response',
    'CachedLLMClient'
]