from googleapiclient.discovery import build
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow

import os, json, base64, re, html, quopri
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv()

SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

def get_gmail_service():
    creds = None

    token_env = os.getenv("GOOGLE_TOKEN")
    if not token_env:
        raise Exception("Missing GOOGLE_TOKEN in .env or Render ENV")

    creds = Credentials.from_authorized_user_info(json.loads(token_env), SCOPES)

    # Nếu token hết hạn thì refresh
    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
        print("Refreshed token:", creds.to_json())  # chỉ để debug log

    return build("gmail", "v1", credentials=creds)


def clean_email_body(body_html):
    try:
        soup = BeautifulSoup(body_html, "html.parser")
        for tag in soup(["script", "style", "img", "button"]):
            tag.decompose()
        for a in soup.find_all("a"):
            a.unwrap()
        text = soup.get_text(separator=" ")
        text = html.unescape(text)
        text = re.sub(r"[\u200B-\u200F\uFEFF]", "", text)
        text = text.replace("\xa0", " ").replace("\u2060", "").replace("\ufeff", "")
        text = text.replace("&zwnj;", "")
        text = re.sub(r"\s+", " ", text).strip()
        return text
    except Exception:
        return body_html


def get_body_from_parts(part):
    mimeType = part.get("mimeType", "")
    body_data = part.get("body", {}).get("data")

    if body_data:
        try:
            decoded_bytes = base64.urlsafe_b64decode(body_data.encode("UTF-8"))
            decoded = decoded_bytes.decode("utf-8", errors="ignore")

            if "=20" in decoded or "=3D" in decoded or "=\n" in decoded:
                decoded = quopri.decodestring(decoded).decode("utf-8", errors="ignore")
        except Exception as e:
            print("Decode error:", e)
            decoded = ""

        if "html" in mimeType or "<html" in decoded.lower():
            return clean_email_body(decoded)
        return clean_email_body(decoded)

    if "parts" in part:
        for p in part["parts"]:
            result = get_body_from_parts(p)
            if result:
                return result
    return None


def get_latest_emails(n=5):
    service = get_gmail_service()
    results = service.users().messages().list(userId='me', maxResults=n).execute()
    messages = results.get('messages', [])

    emails = []
    for msg in messages:
        txt = service.users().messages().get(userId='me', id=msg['id'], format="full").execute()
        payload = txt['payload']
        headers = payload['headers']

        subject = sender = date = ""
        for header in headers:
            name = header['name'].lower()
            if name == 'subject':
                subject = header['value']
            elif name == 'from':
                sender = header['value']
            elif name == 'date':
                date = header['value']

        body = get_body_from_parts(payload) or ""

        emails.append({
            "subject": subject,
            "sender": sender,
            "date": date,
            "body": body
        })

    return emails
