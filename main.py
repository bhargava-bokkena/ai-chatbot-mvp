import os
import json
import re
from datetime import datetime
from typing import Any, Dict, Optional, List

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import PlainTextResponse, HTMLResponse
from twilio.twiml.messaging_response import MessagingResponse

from dotenv import load_dotenv
from openai import OpenAI

from db import init_db, log_message, get_recent_messages, get_last_message_for_sender
from emailer import send_handoff_email


# =========================
# Setup
# =========================
load_dotenv()
init_db()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

DASH_TOKEN = os.getenv("DASH_TOKEN", "").strip()

# SendGrid
SENDGRID_API_KEY = os.getenv("SENDGRID_API_KEY", "").strip()
ALERT_EMAIL_TO = os.getenv("ALERT_EMAIL_TO", "").strip()
ALERT_EMAIL_FROM = os.getenv("ALERT_EMAIL_FROM", "").strip()

app = FastAPI()


# =========================
# Business Context
# =========================
BUSINESS_CONTEXT = {
    "business_name": "This business",
    "business_type": "Local service business",
    "location": "UNKNOWN",
    "hours": "UNKNOWN",
    "services": "UNKNOWN",
    "handoff_message": "Thanks for reaching out — I’m going to pass this to the owner and they’ll follow up shortly.",
    "hours_unknown_message": "I don’t have the hours handy right now, but I can check with the owner and get back to you shortly.",
    "location_unknown_message": "I don’t have the address handy right now, but I can check with the owner and follow up shortly.",
    "inventory_unknown_message": "We don’t carry retail products like that. We primarily offer services — would you like help with any of our services?",
    "emergency_message": "If this is an emergency, please contact local emergency services immediately.",
    "short_message_clarify": "Could you please share a bit more detail so I can help?",
    "booking_question_template": "To help with your booking request, what service is it for, what’s your name, and the best contact number?",
    "booking_details_received": "Thanks — got it. I’ll pass this to the owner to confirm availability and follow up shortly.",
}


# =========================
# Regex helpers
# =========================
EMERGENCY_KEYWORDS = re.compile(r"\b(911|emergency|ambulance|police|fire|danger)\b", re.I)
HOURS_QUERY = re.compile(r"\b(open|close|hours|open today)\b", re.I)
LOCATION_QUERY = re.compile(r"\b(address|location|where are you)\b", re.I)
INVENTORY_KEYWORDS = re.compile(r"\b(milk|eggs|bread|toys|gems)\b", re.I)
BOOKING_HINT = re.compile(r"\b(\d{6,}|\d{3}[- ]?\d{3}[- ]?\d{4})\b")


# =========================
# Helpers
# =========================
def now_ts() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def looks_like_question(text: str) -> bool:
    return "?" in text or text.lower().startswith(("what", "which", "when", "how", "can you", "could you", "please"))


def enforce_length(text: str) -> str:
    text = re.sub(r"\s+", " ", (text or "")).strip()
    return text[:240]


def send_alert_if_needed(tags: List[str], needs_handoff: bool, from_number: str, incoming: str, reply_text: str) -> None:
    """
    Sends an email alert when we need owner follow-up.
    Logs SendGrid status/errors to Render logs.
    Never breaks the user reply flow if email fails.
    """
    if not needs_handoff:
        return

    important = {"booking", "hours", "location", "complaint", "emergency"}
    if not any(t in important for t in tags):
        return

    subject = f"[Handoff] {BUSINESS_CONTEXT['business_name']} – {', '.join(tags)}"
    body = (
        f"From: {from_number}\n"
        f"Tags: {', '.join(tags)}\n\n"
        f"Customer message:\n{incoming}\n\n"
        f"Reply sent:\n{reply_text}\n"
    )

    try:
        status = send_handoff_email(
            subject=subject,
            content=body,
            to_email=ALERT_EMAIL_TO,
            from_email=ALERT_EMAIL_FROM,
            api_key=SENDGRID_API_KEY,
        )
        print("SendGrid alert send attempt. status=", status)
    except Exception as e:
        print("SendGrid: exception:", repr(e))


# =========================
# Routes
# =========================
@app.get("/")
def root():
    return {"ok": True}


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/logs")
def logs(token: str = ""):
    if token != DASH_TOKEN:
        raise HTTPException(status_code=401)

    rows = get_recent_messages()
    html = ["<html><body><table border=1>"]
    html.append("<tr><th>Time</th><th>From</th><th>Inbound</th><th>Reply</th><th>Tags</th><th>Handoff</th></tr>")
    for r in rows:
        html.append(
            f"<tr><td>{r['ts']}</td><td>{r['from_number']}</td>"
            f"<td>{(r['inbound_text'] or '').replace('<','&lt;')}</td>"
            f"<td>{(r['reply_text'] or '').replace('<','&lt;')}</td>"
            f"<td>{r['tags']}</td><td>{r['needs_handoff']}</td></tr>"
        )
    html.append("</table></body></html>")
    return HTMLResponse("".join(html))


@app.post("/webhook/inbound")
async def inbound(request: Request):
    form = await request.form()
    incoming = (form.get("Body") or "").strip()
    from_number = (form.get("From") or "").strip()
    to_number = (form.get("To") or "").strip()

    ts = now_ts()

    reply_text = BUSINESS_CONTEXT["handoff_message"]
    needs_handoff = True
    tags: List[str] = ["general"]

    last = get_last_message_for_sender(from_number)
    last_reply = (last.get("reply_text") if last else "") or ""
    last_tags = ((last.get("tags") if last else "") or "").split(",") if last else []

    # Follow-up booking details
    if "booking" in last_tags and looks_like_question(last_reply):
        if BOOKING_HINT.search(incoming) or len(incoming.split()) >= 2:
            reply_text = BUSINESS_CONTEXT["booking_details_received"]
            tags = ["booking"]
            needs_handoff = True

            reply_text = enforce_length(reply_text)

            log_message(ts, "whatsapp", from_number, to_number, incoming, reply_text, tags, needs_handoff)
            send_alert_if_needed(tags, needs_handoff, from_number, incoming, reply_text)

            resp = MessagingResponse()
            resp.message(reply_text)
            return PlainTextResponse(str(resp), media_type="application/xml")

    # Short message
    if len(incoming) < 3:
        reply_text = BUSINESS_CONTEXT["short_message_clarify"]
        needs_handoff = False
        tags = ["general"]

    elif EMERGENCY_KEYWORDS.search(incoming):
        reply_text = BUSINESS_CONTEXT["emergency_message"]
        tags = ["emergency"]
        needs_handoff = True

    elif INVENTORY_KEYWORDS.search(incoming):
        reply_text = BUSINESS_CONTEXT["inventory_unknown_message"]
        tags = ["service_question"]
        needs_handoff = False

    elif HOURS_QUERY.search(incoming):
        reply_text = BUSINESS_CONTEXT["hours_unknown_message"]
        tags = ["hours"]
        needs_handoff = True

    elif LOCATION_QUERY.search(incoming):
        reply_text = BUSINESS_CONTEXT["location_unknown_message"]
        tags = ["location"]
        needs_handoff = True

    else:
        if client:
            try:
                completion = client.chat.completions.create(
                    model="gpt-4.1-mini",
                    messages=[
                        {"role": "system", "content": "You reply for a local business. Keep replies short and safe."},
                        {"role": "user", "content": incoming},
                    ],
                    temperature=0.2,
                )
                reply_text = (completion.choices[0].message.content or "").strip()
                needs_handoff = False
                tags = ["general"]
            except Exception as e:
                print("OpenAI error:", repr(e))
                reply_text = BUSINESS_CONTEXT["handoff_message"]
                needs_handoff = True
                tags = ["other"]

    reply_text = enforce_length(reply_text)

    # Booking question override
    if "book" in incoming.lower():
        reply_text = BUSINESS_CONTEXT["booking_question_template"]
        tags = ["booking"]
        needs_handoff = True

    log_message(ts, "whatsapp", from_number, to_number, incoming, reply_text, tags, needs_handoff)
    send_alert_if_needed(tags, needs_handoff, from_number, incoming, reply_text)

    resp = MessagingResponse()
    resp.message(reply_text)
    return PlainTextResponse(str(resp), media_type="application/xml")

