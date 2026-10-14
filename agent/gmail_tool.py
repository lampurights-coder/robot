"""LangChain tool: send email from your personal Gmail."""

import base64
from email.mime.text import MIMEText
from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from langchain_core.tools import tool

SCOPES = ["https://www.googleapis.com/auth/gmail.send"]
ROOT = Path(__file__).resolve().parent
TOKEN = ROOT / "token.json"


def _gmail_service():
    if not TOKEN.exists():
        raise RuntimeError(
            "Missing token.json. Run once: python gmail_auth.py"
        )
    creds = Credentials.from_authorized_user_file(str(TOKEN), SCOPES)
    if creds.expired and creds.refresh_token:
        creds.refresh(Request())
        TOKEN.write_text(creds.to_json(), encoding="utf-8")
    return build("gmail", "v1", credentials=creds)


@tool
def send_gmail(to: str, subject: str, body: str) -> str:
    """Send an email from my personal Gmail account.

    Args:
        to: Recipient email address
        subject: Email subject line
        body: Plain-text email body
    """
    service = _gmail_service()
    message = MIMEText(body)
    message["to"] = to
    message["subject"] = subject
    raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
    sent = (
        service.users()
        .messages()
        .send(userId="me", body={"raw": raw})
        .execute()
    )
    return f"Email sent to {to} (id={sent.get('id')})"
