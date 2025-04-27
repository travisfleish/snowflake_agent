import os
from dotenv import load_dotenv
import requests
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.parse
import json
import snowflake.connector
import base64
import pandas as pd

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

                # Test Snowflake connection with token
                try:
                    # Create connection
                    conn = snowflake.connector.connect(
                        account=account,
                        authenticator='oauth',
                        token=access_token,
                        database=database,
                        schema=schema,
                        warehouse=warehouse
                    )

                    # Get list of tables
                    cursor = conn.cursor()
                    cursor.execute(f"SHOW TABLES IN {database}.{schema}")
                    tables = cursor.fetchall()
                    cursor.close()

                    if not tables:
                        error_message = """
                        <html>
                        <body>
                            <h1>No Tables Found</h1>
                            <p>No tables were found in the specified database and schema.</p>
                        </body>
                        </html>
                        """
                        self.wfile.write(error_message.encode())
                        print("No tables found in database")
                        return

                    # Select a table (first one)
                    table_name = tables[0][1]  # column 1 usually contains table name

                    # Fetch 10 random rows
                    cursor = conn.cursor()
                    query = f"SELECT * FROM {database}.{schema}.{table_name} ORDER BY RANDOM() LIMIT 10"
                    print(f"Executing query: {query}")
                    cursor.execute(query)

                    # Fetch column names
                    column_names = [desc[0] for desc in cursor.description]

                    # Fetch data and convert to DataFrame
                    data = cursor.fetchall()
                    df = pd.DataFrame(data, columns=column_names)

                    # Close cursor and connection
                    cursor.close()
                    conn.close()

                    # Convert DataFrame to HTML for display
                    df_html = df.to_html(index=False)

                    success_message = f"""
                    <html>
                    <body>
                        <h1>Query Successful!</h1>
                        <p>Successfully connected to Snowflake with OAuth.</p>
                        <p>Retrieved 10 random rows from {table_name}:</p>
                        {df_html}
                    </body>
                    </html>
                    """
                    self.wfile.write(success_message.encode())

                    # Print to console
                    print(f"Connection successful! Retrieved data from {table_name}")
                    print(df)

                except Exception as e:
                    error_message = f"""
                    <html>
                    <body>
                        <h1>Snowflake Query Error</h1>
                        <p>Error: {str(e)}</p>
                    </body>
                    </html>
                    """
                    self.wfile.write(error_message.encode())
                    print(f"Snowflake query error: {str(e)}")
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