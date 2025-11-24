import os.path
import logging
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

logger = logging.getLogger(__name__)

# If modifying these scopes, delete the file token.json.
SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]


class GmailTool:
    def __init__(self):
        self.creds = None
        self.service = None
        self._authenticate()

    def _authenticate(self):
        """Handles OAuth2 flow."""
        if os.path.exists("token.json"):
            self.creds = Credentials.from_authorized_user_file("token.json", SCOPES)

        # If there are no (valid) credentials available, let the user log in.
        if not self.creds or not self.creds.valid:
            if self.creds and self.creds.expired and self.creds.refresh_token:
                try:
                    self.creds.refresh(Request())
                except Exception as e:
                    logger.error(f"Error refreshing token: {e}")
                    self.creds = None

            if not self.creds:
                # Check for client_secret.json (standard name) or credentials.json
                secret_file = "client_secret.json"
                if not os.path.exists(secret_file):
                    secret_file = "credentials.json"

                if os.path.exists(secret_file):
                    try:
                        flow = InstalledAppFlow.from_client_secrets_file(
                            secret_file, SCOPES
                        )
                        self.creds = flow.run_local_server(port=0)
                        # Save the credentials for the next run
                        with open("token.json", "w") as token:
                            token.write(self.creds.to_json())
                    except Exception as e:
                        logger.error(f"OAuth flow failed: {e}")
                else:
                    logger.warning(
                        "client_secret.json or credentials.json not found. Gmail access disabled."
                    )

        if self.creds:
            self.service = build("gmail", "v1", credentials=self.creds)

    def get_relevant_emails(self, query_str, max_results=5):
        """Searches Gmail for the query string."""
        if not self.service:
            return [
                "System: Gmail access is not configured. Please add client_secret.json."
            ]

        try:
            # Call the Gmail API
            results = (
                self.service.users()
                .messages()
                .list(userId="me", q=query_str, maxResults=max_results)
                .execute()
            )
            messages = results.get("messages", [])

            email_texts = []
            if not messages:
                return ["System: No matching emails found in your Gmail."]

            for message in messages:
                msg = (
                    self.service.users()
                    .messages()
                    .get(userId="me", id=message["id"])
                    .execute()
                )
                snippet = msg.get("snippet", "")
                headers = msg.get("payload", {}).get("headers", [])
                subject = next(
                    (h["value"] for h in headers if h["name"] == "Subject"),
                    "No Subject",
                )
                sender = next(
                    (h["value"] for h in headers if h["name"] == "From"), "Unknown"
                )

                email_texts.append(
                    f"Email From: {sender} | Subject: {subject} | Content: {snippet}"
                )

            return email_texts

        except Exception as e:
            logger.error(f"Gmail API error: {e}")
            return [f"System: Error accessing Gmail: {str(e)}"]
