"""
Configuration settings for the Snowflake Agent application.
Loads environment variables from .env file and provides access to them.
"""

import os
import logging
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Configure logger
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()


class SnowflakeConfig:
    """Snowflake connection configuration settings."""

    def __init__(self):
        self.account = os.getenv("SNOWFLAKE_ACCOUNT")
        self.user = os.getenv("SNOWFLAKE_USER")
        self.password = os.getenv("SNOWFLAKE_PASSWORD")
        self.database = os.getenv("SNOWFLAKE_DATABASE")
        self.schema = os.getenv("SNOWFLAKE_SCHEMA")
        self.warehouse = os.getenv("SNOWFLAKE_WAREHOUSE")
        self.role = os.getenv("SNOWFLAKE_ROLE")

        # Optional parameters
        self.region = os.getenv("SNOWFLAKE_REGION")
        self.private_key_path = os.getenv("SNOWFLAKE_PRIVATE_KEY_PATH")
        self.private_key_passphrase = os.getenv("SNOWFLAKE_PRIVATE_KEY_PASSPHRASE")

    def validate(self) -> bool:
        """
        Validate required Snowflake configuration parameters.

        Returns:
            bool: True if all required parameters are present, False otherwise.
        """
        required_params = ["account", "user"]

        # Either password or private_key_path must be provided
        if not self.password and not self.private_key_path:
            logger.error("Either SNOWFLAKE_PASSWORD or SNOWFLAKE_PRIVATE_KEY_PATH must be provided")
            return False

        for param in required_params:
            if not getattr(self, param):
                logger.error(f"Missing required Snowflake parameter: {param}")
                return False

        return True

    def get_connection_params(self) -> Dict[str, Any]:
        """
        Get Snowflake connection parameters as a dictionary.

        Returns:
            Dict[str, Any]: Dictionary of connection parameters.
        """
        params = {
            "account": self.account,
            "user": self.user,
            "database": self.database,
            "schema": self.schema,
            "warehouse": self.warehouse,
            "role": self.role
        }

        # Filter out None values
        params = {k: v for k, v in params.items() if v is not None}

        # Add authentication
        if self.password:
            params["password"] = self.password
        elif self.private_key_path:
            params["private_key_path"] = self.private_key_path
            if self.private_key_passphrase:
                params["private_key_passphrase"] = self.private_key_passphrase

        # Add region if specified
        if self.region:
            params["region"] = self.region

        return params


class OpenAIConfig:
    """OpenAI API configuration settings."""

    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.organization = os.getenv("OPENAI_ORGANIZATION")
        self.model_name = os.getenv("MODEL_NAME", "gpt-4-turbo")
        self.max_tokens = int(os.getenv("MAX_TOKENS", "4096"))
        self.temperature = float(os.getenv("TEMPERATURE", "0.7"))

    def validate(self) -> bool:
        """
        Validate required OpenAI configuration parameters.

        Returns:
            bool: True if all required parameters are present, False otherwise.
        """
        if not self.api_key:
            logger.error("Missing required OpenAI parameter: api_key")
            return False

        return True

    def get_client_params(self) -> Dict[str, Any]:
        """
        Get OpenAI client parameters as a dictionary.

        Returns:
            Dict[str, Any]: Dictionary of client parameters.
        """
        params = {
            "api_key": self.api_key,
        }

        if self.organization:
            params["organization"] = self.organization

        return params

    def get_completion_params(self) -> Dict[str, Any]:
        """
        Get OpenAI completion parameters as a dictionary.

        Returns:
            Dict[str, Any]: Dictionary of completion parameters.
        """
        return {
            "model": self.model_name,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature
        }


class CrewAIConfig:
    """CrewAI configuration settings."""

    def __init__(self):
        self.api_key = os.getenv("CREWAI_API_KEY")

    def validate(self) -> bool:
        """
        Validate CrewAI configuration parameters.

        Returns:
            bool: True if API key is present when using CrewAI API, False otherwise.
        """
        # CrewAI API key is optional depending on how you're using CrewAI
        return True


class Settings:
    """Main settings container for the application."""

    def __init__(self):
        self.snowflake = SnowflakeConfig()
        self.openai = OpenAIConfig()
        self.crewai = CrewAIConfig()

        # Application settings
        self.debug_mode = os.getenv("DEBUG_MODE", "False").lower() == "true"
        self.log_level = os.getenv("LOG_LEVEL", "INFO")

    def validate(self) -> bool:
        """
        Validate all configuration settings.

        Returns:
            bool: True if all configurations are valid, False otherwise.
        """
        return (
                self.snowflake.validate()
                and self.openai.validate()
                and self.crewai.validate()
        )


# Create a singleton instance
settings = Settings()

# Export the singleton
__all__ = ["settings"]