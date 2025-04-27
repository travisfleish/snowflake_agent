import os
from dotenv import load_dotenv
import requests
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.parse
import json
import snowflake.connector
import base64

# Load environment variables
load_dotenv()

# OAuth configuration
client_id = os.getenv("SNOWFLAKE_OAUTH_CLIENT_ID")
client_secret = os.getenv("SNOWFLAKE_OAUTH_CLIENT_SECRET")
redirect_uri = os.getenv("SNOWFLAKE_OAUTH_REDIRECT_URI")
authorize_url = os.getenv("SNOWFLAKE_OAUTH_AUTHORIZE_ENDPOINT")
token_url = os.getenv("SNOWFLAKE_OAUTH_TOKEN_ENDPOINT")

# Snowflake configuration
account = os.getenv("SNOWFLAKE_ACCOUNT")
database = os.getenv("SNOWFLAKE_DATABASE")
schema = os.getenv("SNOWFLAKE_SCHEMA")
warehouse = os.getenv("SNOWFLAKE_WAREHOUSE")


# Function to decode JWT token
def decode_jwt(token):
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


# Generate authorization URL
auth_params = {
    "client_id": client_id,
    "redirect_uri": redirect_uri,
    "response_type": "code",
    "scope": "session:role-any",
    "state": "somestate"  # In production, use a secure random state
}

auth_url = f"{authorize_url}?{urllib.parse.urlencode(auth_params)}"

# Open browser to authentication URL
print(f"Opening browser to: {auth_url}")
webbrowser.open(auth_url)


# Create a simple HTTP server to receive the callback
class OAuthCallbackHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()

        # Parse query parameters
        query = urllib.parse.urlparse(self.path).query
        params = urllib.parse.parse_qs(query)

        if 'code' in params:
            auth_code = params['code'][0]
            print(f"Received authorization code: {auth_code}")

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

                # Decode and display token payload
                token_payload = decode_jwt(access_token)
                print("Full token payload:")
                print(json.dumps(token_payload, indent=2))
                print("\nSub value:", token_payload.get('sub'))

                # Test Snowflake connection with token
                try:
                    conn = snowflake.connector.connect(
                        account=account,
                        authenticator='oauth',
                        token=access_token,
                        database=database,
                        schema=schema,
                        warehouse=warehouse,
                        log_level='DEBUG'  # Add debug logging
                    )

                    cursor = conn.cursor()
                    cursor.execute("SELECT current_user(), current_role()")
                    result = cursor.fetchone()
                    cursor.close()
                    conn.close()

                    success_message = f"""
                    <html>
                    <body>
                        <h1>Authentication Successful!</h1>
                        <p>Successfully connected to Snowflake with OAuth.</p>
                        <p>User: {result[0]}</p>
                        <p>Role: {result[1]}</p>
                    </body>
                    </html>
                    """
                    self.wfile.write(success_message.encode())
                    print(f"Connection successful! User: {result[0]}, Role: {result[1]}")

                except Exception as e:
                    error_message = f"""
                    <html>
                    <body>
                        <h1>Snowflake Connection Error</h1>
                        <p>Error: {str(e)}</p>
                        <p>Your sub value from the token is: {token_payload.get('sub')}</p>
                        <p>This value needs to match a login_name in Snowflake.</p>
                    </body>
                    </html>
                    """
                    self.wfile.write(error_message.encode())
                    print(f"Snowflake connection error: {str(e)}")
            else:
                error_message = f"""
                <html>
                <body>
                    <h1>Token Exchange Error</h1>
                    <p>Status: {token_response.status_code}</p>
                    <p>Response: {token_response.text}</p>
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
                <p>No authorization code received</p>
            </body>
            </html>
            """
            self.wfile.write(error_message.encode())
            print("No authorization code received")


# Parse the redirect URI to get host and port
parsed_uri = urllib.parse.urlparse(redirect_uri)
server_address = (
    parsed_uri.hostname or 'localhost',
    parsed_uri.port or 8080
)

# Start the server
httpd = HTTPServer(server_address, OAuthCallbackHandler)
print(f"Starting OAuth callback server at {server_address[0]}:{server_address[1]}")
httpd.handle_request()  # Handle one request then exit