import os
import re
from datetime import datetime
from typing import List

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

    # ✅ Updated wording
    "booking_question_template": "To help with your booking request, please provide the service it’s for, your name, and the best contact number.",
    "booking_details_received": "Thanks — got it. I’ll pass this to the owner to confirm availability and follow up shortly.",
}


# =========================
# Regex helpers
# =========================
EMERGENCY_KEYWORDS = re.compile(r"\b(911|emergency|ambulance|police|fire|danger)\b", re.I)
HOURS_QUERY = re.compile(r"\b(open|close|hours|open today|open now)\b", re.I)
LOCATION_QUERY = re.compile(r"\b(address|location|where are you|directions)\b", re.I)
INVENTORY_KEYWORDS = re.compile(r"\b(milk|eggs|bread|toys|gems)\b", re.I)

# Phone-like or long digit strings often indicate contact info
BOOKING_DETAILS_HINT = re.compile(r"\b(\d{6,}|\d{3}[- ]?\d{3}[- ]?\d{4})\b")


# =========================
# Helpers
# =========================
def now_ts() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def looks_like_question(text: str) -> bool:
    t = (text or "").strip().lower()
    return "?" in t or t.startswith(("what", "which", "when", "how", "can you", "could you", "please"))


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def enforce_length(text: str, max_len: int = 240) -> str:
    t = clean_text(text)
    return t[:max_len]


def send_alert_if_needed(tags: List[str], needs_handoff: bool, from_number: str, incoming: str, reply_text: str) -> None:
    """
    Sends an email alert when owner follow-up is needed.
    Quiet on success; logs only on errors.
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
        # No-op if env vars missing (emailer returns None), still safe
        send_handoff_email(
            subject=subject,
            content=body,
            to_email=ALERT_EMAIL_TO,
            from_email=ALERT_EMAIL_FROM,
            api_key=SENDGRID_API_KEY,
        )
    except Exception as e:
        # Only log real failures; never break user reply flow
        print("SendGrid alert error:", repr(e), flush=True)


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
    if not DASH_TOKEN:
        raise HTTPException(status_code=404)
    if token != DASH_TOKEN:
        raise HTTPException(status_code=401)

    rows = get_recent_messages(limit=200)
    html = ["<html><body><h2>Recent Messages</h2><table border=1 cellpadding=6 cellspacing=0>"]
    html.append("<tr><th>Time</th><th>From</th><th>Inbound</th><th>Reply</th><th>Tags</th><th>Handoff</th></tr>")
    for r in rows:
        inbound = (r.get("inbound_text") or "").replace("<", "&lt;")
        reply = (r.get("reply_text") or "").replace("<", "&lt;")
        html.append(
            f"<tr><td>{r.get('ts')}</td><td>{r.get('from_number')}</td>"
            f"<td>{inbound}</td><td>{reply}</td><td>{r.get('tags')}</td><td>{r.get('needs_handoff')}</td></tr>"
        )
    html.append("</table></body></html>")
    return HTMLResponse("".join(html))


@app.post("/webhook/inbound")
async def inbound(request: Request):
    form = await request.form()
    incoming = clean_text(form.get("Body") or "")
    from_number = clean_text(form.get("From") or "")
    to_number = clean_text(form.get("To") or "")

    ts = now_ts()

    # Defaults
    reply_text = BUSINESS_CONTEXT["handoff_message"]
    needs_handoff = True
    tags: List[str] = ["general"]

    # --------- Minimal booking follow-up memory ----------
    last = get_last_message_for_sender(from_number)
    last_reply = clean_text((last.get("reply_text") if last else "") or "")
    last_tags_str = (last.get("tags") if last else "") or ""
    last_tags = [t.strip() for t in last_tags_str.split(",") if t.strip()]

    last_was_booking_question = ("booking" in last_tags) and looks_like_question(last_reply)

    # If last message asked for booking details, and user now sends details, acknowledge & handoff
    if last_was_booking_question and incoming:
        if BOOKING_DETAILS_HINT.search(incoming) or len(incoming.split()) >= 2:
            reply_text = BUSINESS_CONTEXT["booking_details_received"]
            tags = ["booking"]
            needs_handoff = True

            reply_text = enforce_length(reply_text)
            log_message(ts, "whatsapp", from_number, to_number, incoming, reply_text, tags, needs_handoff)
            send_alert_if_needed(tags, needs_handoff, from_number, incoming, reply_text)

            resp = MessagingResponse()
            resp.message(reply_text)
            return PlainTextResponse(str(resp), media_type="application/xml")

    # --------- Rule-based shortcuts ----------
    if len(incoming) < 3:
        reply_text = BUSINESS_CONTEXT["short_message_clarify"]
        needs_handoff = False
        tags = ["general"]

    elif EMERGENCY_KEYWORDS.search(incoming):
        reply_text = BUSINESS_CONTEXT["emergency_message"]
        needs_handoff = True
        tags = ["emergency"]

    elif INVENTORY_KEYWORDS.search(incoming):
        reply_text = BUSINESS_CONTEXT["inventory_unknown_message"]
        needs_handoff = False
        tags = ["service_question"]

    elif HOURS_QUERY.search(incoming):
        reply_text = BUSINESS_CONTEXT["hours_unknown_message"]
        needs_handoff = True
        tags = ["hours"]

    elif LOCATION_QUERY.search(incoming):
        reply_text = BUSINESS_CONTEXT["location_unknown_message"]
        needs_handoff = True
        tags = ["location"]

    # --------- Booking override (deterministic) ----------
    # Trigger booking flow when user asks to book/schedule/appointment
    booking_trigger = any(w in incoming.lower() for w in ["book", "booking", "schedule", "appointment"])
    if booking_trigger:
        reply_text = BUSINESS_CONTEXT["booking_question_template"]
        needs_handoff = True
        tags = ["booking"]

    # --------- LLM fallback ----------
    if tags == ["general"] and client and incoming:
        try:
            completion = client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[
                    {"role": "system", "content": "You reply for a local service business. Be short, professional, and safe."},
                    {"role": "user", "content": incoming},
                ],
                temperature=0.2,
            )
            reply_text = clean_text((completion.choices[0].message.content or "").strip())
            needs_handoff = False
            tags = ["general"]
        except Exception as e:
            print("OpenAI error:", repr(e), flush=True)
            reply_text = BUSINESS_CONTEXT["handoff_message"]
            needs_handoff = True
            tags = ["other"]

    reply_text = enforce_length(reply_text)

    log_message(ts, "whatsapp", from_number, to_number, incoming, reply_text, tags, needs_handoff)
    send_alert_if_needed(tags, needs_handoff, from_number, incoming, reply_text)

    resp = MessagingResponse()
    resp.message(reply_text)
    return PlainTextResponse(str(resp), media_type="application/xml")

