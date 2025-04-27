import os
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# OAuth Configuration
client_id = os.getenv("SNOWFLAKE_OAUTH_CLIENT_ID")
client_secret = os.getenv("SNOWFLAKE_OAUTH_CLIENT_SECRET")
token_url = os.getenv("SNOWFLAKE_OAUTH_TOKEN_ENDPOINT")

def fetch_oauth_token() -> str:
    """
    Fetch an OAuth access token from Okta using client credentials flow.

    Returns:
        str: Access token
    """
    if not all([client_id, client_secret, token_url]):
        raise ValueError("Missing OAuth configuration in environment variables.")

    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "Authorization": f"Basic {encode_credentials(client_id, client_secret)}"
    }

    data = {
        "grant_type": "client_credentials",
        "scope": "session:role-any"  # Or whatever scopes you need
    }

    response = requests.post(token_url, headers=headers, data=data)

    if response.status_code != 200:
        raise Exception(f"Failed to fetch token: {response.status_code} {response.text}")

    token_data = response.json()
    access_token = token_data.get("access_token")

    if not access_token:
        raise Exception("No access_token found in OAuth response.")

    return access_token

def encode_credentials(client_id: str, client_secret: str) -> str:
    """
    Encode client_id and client_secret for basic auth.

    Returns:
        str: Base64 encoded credentials
    """
    import base64
    creds = f"{client_id}:{client_secret}"
    return base64.b64encode(creds.encode()).decode()

if __name__ == "__main__":
    try:
        token = fetch_oauth_token()
        print(f"✅ Access Token:\n{token}\n")

        # Optionally: initialize your connector immediately after fetching
        from utils.snowflake_connector import init_with_oauth_token
        init_with_oauth_token(token)
        print("✅ Snowflake connector initialized with OAuth token!")

    except Exception as e:
        print(f"❌ Error fetching token: {e}")
