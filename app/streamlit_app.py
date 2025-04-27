"""
Streamlit application for the Snowflake Agent.
Provides the user interface and handles UI interactions.
"""

import streamlit as st
import asyncio
from datetime import datetime

from config.settings import settings
from app.connectors import lazy_connector
from app.crew_manager import initialize_crew, run_crew_analysis

def run_streamlit_app():
    """Main function to run the Streamlit application."""

    # Configure Streamlit page
    st.set_page_config(
        page_title="Snowflake Agent",
        page_icon="❄️",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # App title and description
    st.title("❄️ Snowflake Agent")
    st.markdown("""
    Use AI agents to analyze Snowflake data and get business insights.
    Enter your business question and let the agents do the work!
    """)

    # Sidebar for configuration
    sidebar_config = create_sidebar()

    # Main input area
    main_input = create_main_input()

    # Process form submission
    if st.button("Submit", type="primary"):
        process_submission(main_input, sidebar_config)

def create_sidebar():
    """Create and return sidebar configuration."""
    st.sidebar.title("Configuration")

    # Connection status indicator
    is_configured = settings.validate()

    if is_configured:
        st.sidebar.success("✅ Settings loaded successfully.")
    else:
        st.sidebar.warning("⚠️ Some settings may be missing or invalid. Check your .env file.")

    # OAuth setup button
    if st.sidebar.button("Setup OAuth Authentication"):
        with st.spinner("Setting up OAuth authentication..."):
            success = lazy_connector.setup_oauth()
            if success:
                st.sidebar.success("✅ OAuth authentication successful!")
                # Force rerun to update UI
                st.rerun()
            else:
                st.sidebar.error("❌ OAuth authentication failed.")

    # Crew selection
    crew_type = st.sidebar.selectbox(
        "Select Crew Type",
        ["Business Research", "Sales Analysis"]
    )

    # Advanced options (collapsible)
    with st.sidebar.expander("Advanced Options"):
        verbose_mode = st.checkbox("Verbose Mode", value=False)
        max_results = st.slider("Max Results", min_value=10, max_value=1000, value=100, step=10)

        # Time period selection for Sales Analysis
        time_period = None
        custom_start_date = None
        custom_end_date = None

        if crew_type == "Sales Analysis":
            time_period = st.selectbox(
                "Time Period",
                ["last_month", "last_quarter", "last_year", "custom"]
            )

            if time_period == "custom":
                custom_start_date = st.date_input("Start Date")
                custom_end_date = st.date_input("End Date")

    # Test connection button
    if st.sidebar.button("Test Snowflake Connection"):
        with st.spinner("Testing connection..."):
            if lazy_connector.test_connection():
                st.sidebar.success("✅ Successfully connected to Snowflake!")
            else:
                st.sidebar.error("❌ Failed to connect to Snowflake. Check your credentials or set up OAuth authentication.")

    # Return configuration
    return {
        "crew_type": crew_type,
        "verbose_mode": verbose_mode,
        "max_results": max_results,
        "time_period": time_period,
        "custom_start_date": custom_start_date,
        "custom_end_date": custom_end_date
    }

def create_main_input():
    """Create and return main input area."""
    st.header("Ask a Business Question")
    business_question = st.text_area("Enter your business question", height=100)

    # Additional context (optional)
    context = None
    with st.expander("Additional Context (Optional)"):
        context = st.text_area("Provide additional context for your question", height=100)

    return {
        "business_question": business_question,
        "context": context
    }

def process_submission(main_input, sidebar_config):
    """Process the form submission."""
    business_question = main_input["business_question"]

    if not business_question:
        st.warning("Please enter a business question.")
        return

    # Check if connection is available
    if not lazy_connector.is_available():
        st.error("❌ Snowflake connection is not available. Please set up OAuth authentication or check your configuration.")
        return

    # Initialize progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Run analysis
    run_analysis(business_question, main_input["context"], sidebar_config, progress_bar, status_text)

def run_analysis(question, context, config, progress_bar, status_text):
    """Run the analysis in async mode."""

    # Update status
    status_text.text("Initializing agents...")
    progress_bar.progress(10)

    # Initialize crew
    try:
        crew = initialize_crew(
            crew_type=config["crew_type"],
            verbose=config["verbose_mode"],
            time_period=config.get("time_period"),
            custom_start_date=config.get("custom_start_date"),
            custom_end_date=config.get("custom_end_date")
        )

        # Update status
        status_text.text("Analyzing question and retrieving data...")
        progress_bar.progress(30)

        # Run crew analysis (in a way that works with Streamlit)
        result = asyncio.run(run_crew_analysis(
            crew=crew,
            crew_type=config["crew_type"],
            question=question,
            context=context
        ))

        # Update status
        status_text.text("Analysis complete!")
        progress_bar.progress(100)

        # Display results
        display_results(result, config["crew_type"])

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        import traceback
        st.code(traceback.format_exc(), language="python")

def display_results(result, crew_type):
    """Display the crew results in a structured format."""
    st.header("Results")

    # Create tabs for different result sections
    tabs = st.tabs(["Summary", "Data Analysis", "Insights", "Recommendations", "Raw Data"])

    # Summary tab
    with tabs[0]:
        if crew_type == "Business Research":
            st.markdown(f"### Question: {result.get('question', {}).get('question', 'N/A')}")
            if result.get('answer'):
                st.markdown(result['answer'])
            elif result.get('raw_result'):
                st.markdown(result['raw_result'])
        else:  # Sales Analysis
            if result.get('recommendations'):
                st.markdown(result['recommendations'])
            elif result.get('raw_result'):
                st.markdown(result['raw_result'])

    # Data Analysis tab
    with tabs[1]:
        if crew_type == "Business Research":
            if result.get('data_retrieval'):
                st.markdown("### Data Analysis")
                st.markdown(result['data_retrieval'])
        else:  # Sales Analysis
            if result.get('analysis'):
                st.markdown("### Sales Analysis")
                st.markdown(result['analysis'])

    # Insights tab
    with tabs[2]:
        if crew_type == "Business Research":
            if result.get('analysis'):
                st.markdown("### Insights")
                st.markdown(result['analysis'])
        else:  # Sales Analysis
            if result.get('insights', {}).get('summary'):
                st.markdown("### Key Insights")
                st.markdown(result['insights']['summary'])

    # Recommendations tab
    with tabs[3]:
        if crew_type == "Business Research":
            if result.get('interpretation'):
                st.markdown("### Recommendations & Next Steps")
                st.markdown(result['interpretation'])
        else:  # Sales Analysis
            if result.get('recommendations'):
                st.markdown("### Recommendations")
                st.markdown(result['recommendations'])

    # Raw Data tab
    with tabs[4]:
        st.markdown("### Raw Result Data")
        st.json(result)

    # Add timestamp
    st.markdown(f"*Analysis completed on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")