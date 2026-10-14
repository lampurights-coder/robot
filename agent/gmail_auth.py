"""One-time Gmail OAuth login. Run once to create token.json."""

from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow

SCOPES = ["https://www.googleapis.com/auth/gmail.send"]
ROOT = Path(__file__).resolve().parent
CREDENTIALS = ROOT / "credentials.json"
TOKEN = ROOT / "token.json"


def main() -> None:
    if not CREDENTIALS.exists():
        raise SystemExit(
            f"Missing {CREDENTIALS}\n"
            "Download OAuth Desktop client JSON from Google Cloud and save it there."
        )

    creds = None
    if TOKEN.exists():
        creds = Credentials.from_authorized_user_file(str(TOKEN), SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(str(CREDENTIALS), SCOPES)
            # Opens browser for Google login
            creds = flow.run_local_server(port=0)
        TOKEN.write_text(creds.to_json(), encoding="utf-8")
        print(f"Saved login token → {TOKEN}")
    else:
        print(f"Already authorized → {TOKEN}")

    print("Gmail send auth OK. You can use the send_gmail tool now.")


if __name__ == "__main__":
    main()
