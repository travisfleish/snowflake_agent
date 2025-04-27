"""
Logging configuration for the Snowflake Agent application.
Sets up console and file logging with customizable log levels.
"""

import os
import sys
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

# Default log directory - create logs folder in project root
DEFAULT_LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")

# Create logs directory if it doesn't exist
Path(DEFAULT_LOG_DIR).mkdir(exist_ok=True)


def configure_logging(
        log_level="INFO",
        log_file="snowflake_agent.log",
        log_dir=None,
        max_file_size_mb=10,
        backup_count=5,
):
    """
    Configure application logging to output to both console and file.

    Args:
        log_level (str): Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file (str): Name of the log file
        log_dir (str, optional): Directory to store log files. Defaults to PROJECT_ROOT/logs
        max_file_size_mb (int): Maximum size of log file before rotation in MB
        backup_count (int): Number of backup log files to keep

    Returns:
        logging.Logger: Configured root logger
    """
    if log_dir is None:
        log_dir = DEFAULT_LOG_DIR

    # Create log directory if it doesn't exist
    Path(log_dir).mkdir(exist_ok=True)

    # Set numeric log level
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Clear any existing handlers (in case this is called multiple times)
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    console_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%H:%M:%S'
    )

    # Create file handler for all log levels
    log_file_path = os.path.join(log_dir, log_file)
    file_handler = RotatingFileHandler(
        log_file_path,
        maxBytes=max_file_size_mb * 1024 * 1024,
        backupCount=backup_count
    )
    file_handler.setLevel(numeric_level)
    file_handler.setFormatter(file_formatter)

    # Create console handler with a possibly different log level
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(console_formatter)

    # Add handlers to the root logger
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    # Create a startup message
    root_logger.info(
        f"Logging initialized (level={log_level}, file={log_file_path})"
    )

    return root_logger


def get_logger(name):
    """
    Get a logger with the specified name.

    Args:
        name (str): Logger name, typically __name__ from the calling module

    Returns:
        logging.Logger: Logger with the specified name
    """
    return logging.getLogger(name)


# For direct usage: "from config.logging_config import logger"
logger = get_logger("snowflake_agent")

if __name__ == "__main__":
    # Test logging configuration when this module is run directly
    configure_logging(log_level="DEBUG")
    test_logger = get_logger(__name__)

    test_logger.debug("This is a debug message")
    test_logger.info("This is an info message")
    test_logger.warning("This is a warning message")
    test_logger.error("This is an error message")
    test_logger.critical("This is a critical message")