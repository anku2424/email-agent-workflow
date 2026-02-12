"""Gmail API tools for reading and sending emails."""
import os
import base64
from email.mime.text import MIMEText
from typing import Optional
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build

# Gmail API scopes
SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.compose",  # For creating drafts
    "https://www.googleapis.com/auth/gmail.send",     # For sending flag notifications
    "https://www.googleapis.com/auth/gmail.modify"
]

TOKEN_FILE = "token.json"
CREDENTIALS_FILE = "credentials.json"


def get_gmail_service():
    """Authenticate and return Gmail API service."""
    creds = None
    
    if os.path.exists(TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
    
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not os.path.exists(CREDENTIALS_FILE):
                raise FileNotFoundError(
                    f"Missing {CREDENTIALS_FILE}. Download it from Google Cloud Console."
                )
            flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
            creds = flow.run_local_server(port=0)
        
        with open(TOKEN_FILE, "w") as token:
            token.write(creds.to_json())
    
    return build("gmail", "v1", credentials=creds)


def get_unread_emails(max_results: int = 10) -> list[dict]:
    """Fetch unread emails from Gmail inbox."""
    service = get_gmail_service()
    
    results = service.users().messages().list(
        userId="me",
        labelIds=["INBOX", "UNREAD"],
        maxResults=max_results
    ).execute()
    
    messages = results.get("messages", [])
    emails = []
    
    for msg in messages:
        msg_data = service.users().messages().get(
            userId="me",
            id=msg["id"],
            format="full"
        ).execute()
        
        headers = msg_data["payload"]["headers"]
        subject = next((h["value"] for h in headers if h["name"] == "Subject"), "No Subject")
        sender = next((h["value"] for h in headers if h["name"] == "From"), "Unknown")
        date = next((h["value"] for h in headers if h["name"] == "Date"), "Unknown")
        
        # Extract body
        body = ""
        if "parts" in msg_data["payload"]:
            for part in msg_data["payload"]["parts"]:
                if part["mimeType"] == "text/plain":
                    body = base64.urlsafe_b64decode(part["body"]["data"]).decode("utf-8")
                    break
        elif "body" in msg_data["payload"] and "data" in msg_data["payload"]["body"]:
            body = base64.urlsafe_b64decode(msg_data["payload"]["body"]["data"]).decode("utf-8")
        
        emails.append({
            "id": msg["id"],
            "thread_id": msg["threadId"],
            "subject": subject,
            "sender": sender,
            "date": date,
            "body": body[:2000],  # Truncate long emails
            "snippet": msg_data.get("snippet", "")
        })
    
    return emails


def create_draft(to: str, subject: str, body: str, reply_to_message_id: Optional[str] = None) -> dict:
    """Create a draft email in Gmail (does not send)."""
    service = get_gmail_service()
    
    message = MIMEText(body)
    message["to"] = to
    message["subject"] = subject
    
    raw = base64.urlsafe_b64encode(message.as_bytes()).decode("utf-8")
    
    draft_body = {"message": {"raw": raw}}
    if reply_to_message_id:
        # Get thread ID for proper threading
        original = service.users().messages().get(
            userId="me", id=reply_to_message_id
        ).execute()
        draft_body["message"]["threadId"] = original["threadId"]
    
    result = service.users().drafts().create(userId="me", body=draft_body).execute()
    return {"status": "draft_created", "draft_id": result["id"]}


def send_email(to: str, subject: str, body: str, reply_to_message_id: Optional[str] = None) -> dict:
    """Send an email via Gmail (used only for notifications)."""
    service = get_gmail_service()
    
    message = MIMEText(body)
    message["to"] = to
    message["subject"] = subject
    
    raw = base64.urlsafe_b64encode(message.as_bytes()).decode("utf-8")
    
    body_data = {"raw": raw}
    if reply_to_message_id:
        original = service.users().messages().get(
            userId="me", id=reply_to_message_id
        ).execute()
        body_data["threadId"] = original["threadId"]
    
    result = service.users().messages().send(userId="me", body=body_data).execute()
    return {"status": "sent", "message_id": result["id"]}


def mark_as_read(message_id: str) -> dict:
    """Mark an email as read."""
    service = get_gmail_service()
    service.users().messages().modify(
        userId="me",
        id=message_id,
        body={"removeLabelIds": ["UNREAD"]}
    ).execute()
    return {"status": "marked_read", "message_id": message_id}
