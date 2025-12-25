import os
import re
import logging
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
# Logging
# =========================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("easyreplies")


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
# Business Context (MVP-safe)
# =========================
BUSINESS_CONTEXT = {
    "business_name": "This business",

    "greeting": "Hello! How can we assist you today?",
    "thanks_reply": "You’re welcome! Let us know if you need any assistance.",
    "short_message_clarify": "Could you please share a bit more detail so I can help?",

    "hours_unknown_message": "I don’t have the hours handy right now, but I can check with the owner and get back to you shortly.",
    "location_unknown_message": "I don’t have the address handy right now, but I can check with the owner and follow up shortly.",

    "service_discovery_message": "We offer a range of services. Could you let me know what you’re looking for so I can help further?",
    "retail_redirect_message": "We don’t carry retail products, but I’d be happy to help with our services. What can I assist you with?",

    "booking_question_template": "To help with your booking request, please provide the service you’d like to book, your name, and the best contact number.",
    "booking_details_received": "Thanks — got it! I’ll pass this to the owner to confirm availability and follow up shortly.",

    "handoff_normal": "Thanks — got it! I’ll pass this to the owner and they’ll follow up shortly.",
    "handoff_complaint": "I’m sorry about that. I’ll pass this to the owner so they can follow up with you shortly.",

    "emergency_message": "If this is an emergency, please contact local emergency services immediately.",
}


# =========================
# Regex / Intent helpers
# =========================
EMERGENCY_KEYWORDS = re.compile(r"\b(911|emergency|ambulance|police|fire|danger)\b", re.I)

HOURS_QUERY = re.compile(r"\b(open|close|hours|open today|open now|available today|available now)\b", re.I)
LOCATION_QUERY = re.compile(r"\b(address|location|where are you|directions)\b", re.I)

SERVICE_DISCOVERY = re.compile(
    r"\b(what services|services do you offer|what do you offer|what do you do|services provided|service list)\b",
    re.I,
)

BOOKING_TRIGGER = re.compile(r"\b(book|booking|schedule|appointment|appt|reserve)\b", re.I)
AVAILABILITY_INTENT = re.compile(r"\b(availability|available)\b", re.I)

MONTHS = r"(jan|january|feb|february|mar|march|apr|april|may|jun|june|jul|july|aug|august|sep|sept|september|oct|october|nov|november|dec|december)"
DATE_TOKEN = re.compile(rf"\b({MONTHS}\s*\d{{1,2}}(st|nd|rd|th)?|\d{{1,2}}(st|nd|rd|th)?|tomorrow|today)\b", re.I)
TIME_TOKEN = re.compile(r"\b(\d{1,2}(:\d{2})?\s*(am|pm)|noon|midnight)\b", re.I)

COMPLAINT_KEYWORDS = re.compile(
    r"\b(refund|return|complain|complaint|bad|terrible|awful|unhappy|angry|upset|frustrated|not happy|issue|problem|charged|scam)\b",
    re.I,
)

RETAIL_INTENT = re.compile(
    r"\b(do you (sell|have)|do u (sell|have)|in stock|available in stock|carry|selling)\b",
    re.I,
)

PHONE_LIKE_STRICT = re.compile(r"\b(\+?\d[\d\-\s]{6,}\d)\b")
DIGITS_7PLUS = re.compile(r"\b\d{7,}\b")


# =========================
# Helpers
# =========================
def now_ts() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def detect_source(message: str) -> str:
    m = (message or "").lower()
    if "via your website" in m or "from your website" in m:
        return "website"
    if "via qr" in m or "scanned" in m:
        return "qr"
    if "referred" in m or "told by" in m:
        return "referral"
    return "unknown"


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def enforce_length(text: str, max_len: int = 240) -> str:
    return clean_text(text)[:max_len]


def is_greeting(text: str) -> bool:
    return clean_text(text).lower() in {"hi", "hello", "hey"}


def is_thanks(text: str) -> bool:
    return clean_text(text).lower() in {"thanks", "thank you", "thx", "ty"}


def looks_like_booking_details(text: str) -> bool:
    t = clean_text(text)
    return bool((DIGITS_7PLUS.search(t) or PHONE_LIKE_STRICT.search(t)) and re.search(r"[A-Za-z]{2,}", t))


def send_alert_if_needed(tags: List[str], needs_handoff: bool, from_number: str, incoming: str, reply_text: str) -> None:
    if not needs_handoff:
        return

    subject = f"[Handoff] {BUSINESS_CONTEXT['business_name']} – {', '.join(tags)}"
    body = f"From: {from_number}\n\nCustomer message:\n{incoming}\n\nReply:\n{reply_text}"

    try:
        send_handoff_email(subject, body, ALERT_EMAIL_TO, ALERT_EMAIL_FROM, SENDGRID_API_KEY)
    except Exception as e:
        print("SendGrid error:", repr(e), flush=True)


def reply_and_log(ts, from_number, to_number, incoming, reply_text, tags, needs_handoff):
    reply_text = enforce_length(reply_text)
    log_message(ts, "whatsapp", from_number, to_number, incoming, reply_text, tags, needs_handoff)
    send_alert_if_needed(tags, needs_handoff, from_number, incoming, reply_text)

    resp = MessagingResponse()
    resp.message(reply_text)
    return PlainTextResponse(str(resp), media_type="application/xml")


# =========================
# Routes
# =========================
@app.post("/webhook/inbound")
async def inbound(request: Request):
    form = await request.form()
    incoming = clean_text(form.get("Body") or "")
    source = detect_source(incoming)
    from_number = clean_text(form.get("From") or "")
    to_number = clean_text(form.get("To") or "")

    logger.info(
        "Inbound WhatsApp message",
        extra={
            "from": from_number,
            "to": to_number,
            "source": source,
            "message": incoming[:200],
        },
    )

    ts = now_ts()

    if incoming and looks_like_booking_details(incoming):
        return reply_and_log(ts, from_number, to_number, incoming, BUSINESS_CONTEXT["booking_details_received"], ["booking"], True)

    if len(incoming) < 3:
        return reply_and_log(ts, from_number, to_number, incoming, BUSINESS_CONTEXT["short_message_clarify"], ["general"], False)

    if is_thanks(incoming):
        return reply_and_log(ts, from_number, to_number, incoming, BUSINESS_CONTEXT["thanks_reply"], ["general"], False)

    if is_greeting(incoming):
        return reply_and_log(ts, from_number, to_number, incoming, BUSINESS_CONTEXT["greeting"], ["general"], False)

    if EMERGENCY_KEYWORDS.search(incoming):
        return reply_and_log(ts, from_number, to_number, incoming, BUSINESS_CONTEXT["emergency_message"], ["emergency"], True)

    if COMPLAINT_KEYWORDS.search(incoming):
        return reply_and_log(ts, from_number, to_number, incoming, BUSINESS_CONTEXT["handoff_complaint"], ["complaint"], True)

    return reply_and_log(ts, from_number, to_number, incoming, BUSINESS_CONTEXT["handoff_normal"], ["other"], True)

