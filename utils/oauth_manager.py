"""
OAuth manager for Snowflake integration.
Handles token acquisition, validation, and refresh.
"""

import os
import time
import requests
import logging
from typing import Dict, Any, Optional
from urllib.parse import urlencode

from config.settings import settings

logger = logging.getLogger(__name__)


class SnowflakeOAuthManager:
    """
    Manages OAuth authentication flow for Snowflake.
    """

    def __init__(self):
        """Initialize OAuth manager with settings."""
        self.client_id = settings.snowflake.oauth_client_id
        self.client_secret = settings.snowflake.oauth_client_secret
        self.redirect_uri = settings.snowflake.oauth_redirect_uri
        self.token_endpoint = settings.snowflake.oauth_token_endpoint
        self.authorize_endpoint = settings.snowflake.oauth_authorize_endpoint

        # Token storage - in production, use a more secure storage
        self.tokens = {}

    def get_authorization_url(self, state: str = None, scope: str = "session:role-any") -> str:
        """
        Get the authorization URL for the OAuth flow.

        Args:
            state: Optional state parameter for CSRF protection
            scope: Requested scope

        Returns:
            str: Authorization URL
        """
        params = {
            "client_id": self.client_id,
            "redirect_uri": self.redirect_uri,
            "response_type": "code",
            "scope": scope
        }

        if state:
            params["state"] = state

        return f"{self.authorize_endpoint}?{urlencode(params)}"

    def exchange_code_for_token(self, code: str) -> Dict[str, Any]:
        """
        Exchange authorization code for access token.

        Args:
            code: Authorization code from callback

        Returns:
            Dict[str, Any]: Token response
        """
        data = {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": self.redirect_uri,
            "client_id": self.client_id,
            "client_secret": self.client_secret
        }

        response = requests.post(self.token_endpoint, data=data)
        if response.status_code != 200:
            logger.error(f"Token exchange failed: {response.text}")
            raise Exception(f"Token exchange failed: {response.status_code}")

        token_data = response.json()

        # Store token with user info
        user_id = self._extract_user_from_token(token_data["access_token"])
        self.tokens[user_id] = {
            "access_token": token_data["access_token"],
            "refresh_token": token_data.get("refresh_token"),
            "expires_at": time.time() + token_data["expires_in"],
            "scope": token_data.get("scope", "")
        }

        # Update Snowflake config with the token
        settings.snowflake.oauth_token = token_data["access_token"]

        return token_data

    def refresh_token(self, user_id: str) -> bool:
        """
        Refresh the access token using the refresh token.

        Args:
            user_id: User identifier

        Returns:
            bool: True if refresh was successful
        """
        if user_id not in self.tokens or "refresh_token" not in self.tokens[user_id]:
            return False

        data = {
            "grant_type": "refresh_token",
            "refresh_token": self.tokens[user_id]["refresh_token"],
            "client_id": self.client_id,
            "client_secret": self.client_secret
        }

        response = requests.post(self.token_endpoint, data=data)
        if response.status_code != 200:
            logger.error(f"Token refresh failed: {response.text}")
            return False

        token_data = response.json()

        # Update stored token
        self.tokens[user_id]["access_token"] = token_data["access_token"]
        self.tokens[user_id]["expires_at"] = time.time() + token_data["expires_in"]
        if "refresh_token" in token_data:
            self.tokens[user_id]["refresh_token"] = token_data["refresh_token"]

        # Update Snowflake config with the new token
        settings.snowflake.oauth_token = token_data["access_token"]

        return True

    def is_token_valid(self, user_id: str) -> bool:
        """
        Check if the stored token is valid.

        Args:
            user_id: User identifier

        Returns:
            bool: True if token is valid
        """
        if user_id not in self.tokens:
            return False

        # Check if token is expired (with 5-minute buffer)
        return self.tokens[user_id]["expires_at"] > (time.time() + 300)

    def get_valid_token(self, user_id: str) -> Optional[str]:
        """
        Get a valid access token, refreshing if necessary.

        Args:
            user_id: User identifier

        Returns:
            Optional[str]: Valid access token or None
        """
        if not self.is_token_valid(user_id):
            if not self.refresh_token(user_id):
                return None

        return self.tokens[user_id]["access_token"]

    def _extract_user_from_token(self, token: str) -> str:
        """
        Extract user ID from JWT token.

        Args:
            token: JWT token

        Returns:
            str: User ID
        """
        # Simple implementation - in production use proper JWT decoding
        import base64
        import json

        try:
            payload = token.split('.')[1]
            # Add padding if needed
            payload += '=' * ((4 - len(payload) % 4) % 4)
            decoded = base64.b64decode(payload)
            claims = json.loads(decoded)
            return claims.get("sub", "unknown")
        except Exception as e:
            logger.error(f"Error decoding token: {str(e)}")
            return "unknown"


# Create singleton instance
oauth_manager = SnowflakeOAuthManager()