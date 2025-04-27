"""
OAuth setup for Snowflake integration.
Handles OAuth authorization flow and token management.
"""

import os
import json
import base64
import webbrowser
import requests
import urllib.parse
from http.server import HTTPServer, BaseHTTPRequestHandler
import asyncio
import threading
from typing import Dict, Any, Optional, Callable

from dotenv import load_dotenv
from config.settings import settings
from utils.snowflake_connector import init_with_oauth_token

# Load environment variables
load_dotenv()

# OAuth configuration
client_id = settings.snowflake.oauth_client_id
client_secret = settings.snowflake.oauth_client_secret
redirect_uri = settings.snowflake.oauth_redirect_uri
authorize_url = settings.snowflake.oauth_authorize_endpoint
token_url = settings.snowflake.oauth_token_endpoint


# Function to decode JWT token
def decode_jwt(token):
    """Decode a JWT token to extract payload information."""
    parts = token.split('.')
    if len(parts) != 3:
        return "Invalid token format"

    # JWT tokens have three parts: header.payload.signature
    # We're interested in the payload (second part)
    payload_base64 = parts[1]

    # Add padding if needed
    padding_needed = len(payload_base64) % 4
    if padding_needed:
        payload_base64 += '=' * (4 - padding_needed)

    try:
        decoded_bytes = base64.b64decode(payload_base64)
        payload = json.loads(decoded_bytes)
        return payload
    except Exception as e:
        return f"Error decoding token: {str(e)}"


class OAuthCallbackHandler(BaseHTTPRequestHandler):
    """HTTP handler for OAuth callback."""

    # Class variable to store the token and callback function
    token_data = None
    callback_fn = None

    def do_GET(self):
        """Handle GET request (OAuth callback)."""
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()

        # Parse query parameters
        query = urllib.parse.urlparse(self.path).query
        params = urllib.parse.parse_qs(query)

        if 'code' in params:
            auth_code = params['code'][0]
            print(f"Received authorization code")

            # Exchange code for token
            token_data = {
                "grant_type": "authorization_code",
                "code": auth_code,
                "redirect_uri": redirect_uri,
                "client_id": client_id,
                "client_secret": client_secret
            }

            token_response = requests.post(token_url, data=token_data)

            if token_response.status_code == 200:
                token_json = token_response.json()
                access_token = token_json.get('access_token')

                print("Successfully obtained access token")

                # Store the token data
                OAuthCallbackHandler.token_data = token_json

                # Display success message
                success_message = """
                <html>
                <body>
                    <h1>Authentication Successful!</h1>
                    <p>Successfully authenticated with Snowflake using OAuth.</p>
                    <p>You can now close this window and return to the application.</p>
                </body>
                </html>
                """
                self.wfile.write(success_message.encode())

                # Call the callback function if provided
                if OAuthCallbackHandler.callback_fn:
                    OAuthCallbackHandler.callback_fn(token_json)

            else:
                error_message = f"""
                <html>
                <body>
                    <h1>Token Exchange Error</h1>
                    <p>Failed to obtain access token.</p>
                    <p>Please close this window and try again.</p>
                </body>
                </html>
                """
                self.wfile.write(error_message.encode())
                print(f"Token exchange error: {token_response.text}")
        else:
            error_message = """
            <html>
            <body>
                <h1>Authorization Error</h1>
                <p>No authorization code received.</p>
                <p>Please close this window and try again.</p>
            </body>
            </html>
            """
            self.wfile.write(error_message.encode())
            print("No authorization code received")


def start_oauth_flow(callback: Optional[Callable[[Dict[str, Any]], None]] = None):
    """
    Start the OAuth authorization flow.

    Args:
        callback: Optional callback function to call with token data
    """
    # Set the callback function
    OAuthCallbackHandler.callback_fn = callback

    # Generate authorization URL
    auth_params = {
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": "session:role-any",
        "state": os.urandom(16).hex()  # Secure random state
    }

    auth_url = f"{authorize_url}?{urllib.parse.urlencode(auth_params)}"

    # Open browser to authentication URL
    print(f"Opening browser to authentication URL")
    webbrowser.open(auth_url)

    # Parse the redirect URI to get host and port
    parsed_uri = urllib.parse.urlparse(redirect_uri)
    server_address = (
        parsed_uri.hostname or 'localhost',
        parsed_uri.port or 8080
    )

    # Start the server in a separate thread
    server = HTTPServer(server_address, OAuthCallbackHandler)
    print(f"Starting OAuth callback server at {server_address[0]}:{server_address[1]}")

    # Run the server in a separate thread
    server_thread = threading.Thread(target=server.handle_request)
    server_thread.daemon = True
    server_thread.start()

    return server_thread


def setup_snowflake_with_oauth():
    """
    Set up Snowflake connection using OAuth.

    Returns:
        bool: True if setup was successful, False otherwise
    """

    def handle_token(token_data):
        """Handle the received token data."""
        if token_data and 'access_token' in token_data:
            # Initialize Snowflake connector with token
            connector = init_with_oauth_token(token_data['access_token'])
            return connector is not None
        return False

    # Start OAuth flow
    server_thread = start_oauth_flow(callback=handle_token)

    # Wait for the server thread to complete
    server_thread.join()

    # Check if token was received
    return OAuthCallbackHandler.token_data is not None and 'access_token' in OAuthCallbackHandler.token_data