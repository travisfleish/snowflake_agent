import streamlit as st
from tools.snowflake_tools import execute_snowflake_query
from utils.snowflake_connector import init_with_oauth_token
from fetch_token import fetch_oauth_token  # ← import the token fetcher you just built

# Fetch token + initialize connector before anything else
token = fetch_oauth_token()
init_with_oauth_token(token)

def main():
    st.title("🧪 Snowflake Query Test")

    st.write("Test running a simple query against your Snowflake database.")

    query = st.text_area(
        "Enter SQL Query",
        value="SELECT CURRENT_USER(), CURRENT_ROLE(), CURRENT_DATABASE(), CURRENT_SCHEMA();",
        height=150
    )

    if st.button("Run Query"):
        try:
            st.info("Executing query...")
            result = execute_snowflake_query(query)
            st.success("Query executed successfully!")

            if isinstance(result, str):
                st.code(result, language="markdown")
            else:
                st.write(result)

        except Exception as e:
            st.error(f"Error executing query: {e}")

if __name__ == "__main__":
    main()
c