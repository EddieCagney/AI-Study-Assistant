import os

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

SCOPES = [
    "https://www.googleapis.com/auth/calendar.readonly",
    "https://www.googleapis.com/auth/calendar.events",
]

# Always resolve paths relative to this file's directory so the app works
# regardless of what the current working directory is.
_HERE = os.path.dirname(os.path.abspath(__file__))
TOKEN_PATH = os.path.join(_HERE, "token.json")
CREDENTIALS_PATH = os.path.join(_HERE, "credentials.json")


def get_credentials():
    """Authenticate and return Google API credentials.

    Handles OAuth2 flow on first run and caches the token for future use.
    Supports both Gmail and Calendar scopes.
    """
    creds = None

    if os.path.exists(TOKEN_PATH):
        creds = Credentials.from_authorized_user_file(TOKEN_PATH, SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_PATH, SCOPES)
            creds = flow.run_local_server(port=0)
        with open(TOKEN_PATH, "w") as token:
            token.write(creds.to_json())

    return creds


def get_gmail_service():
    """Return an authenticated Gmail API service."""
    return build("gmail", "v1", credentials=get_credentials())


def get_calendar_service():
    """Return an authenticated Google Calendar API service."""
    return build("calendar", "v3", credentials=get_credentials())
