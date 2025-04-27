"""
Snowflake Agent Entry Point
Initializes environment and starts the Streamlit application.
"""

import os
import streamlit as st
from dotenv import load_dotenv

from config.logging_config import configure_logging
from app.streamlit_app import run_streamlit_app

# Configure logging
logger = configure_logging(log_level="INFO")

# Load environment variables
load_dotenv()

# Main function
def main():
    # Start the Streamlit app
    run_streamlit_app()

if __name__ == "__main__":
    main()