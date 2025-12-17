import os
import re
from datetime import datetime
from typing import List, Tuple

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
# Business Context (MVP-safe, not per-business yet)
# =========================
BUSINESS_CONTEXT = {
    "business_name": "This business",

    # General
    "greeting": "Hello! How can we assist you today?",
    "thanks_reply": "You’re welcome! Let us know if you need any assistance.",
    "short_message_clarify": "Could you please share a bit more detail so I can help?",

    # Hours / Location (unknown in MVP)
    "hours_unknown_message": "I don’t have the hours handy right now, but I can check with the owner and get back to you shortly.",
    "location_unknown_message": "I don’t have the address handy right now, but I can check with the owner and follow up shortly.",

    # Service discovery (MVP generic – no hard-coded service list yet)
    "service_discovery_message": "We offer a range of services. Could you let me know what you’re looking for so I can help further?",

    # Retail / product questions → redirect to services
    "retail_redirect_message": "We don’t carry retail products, but I’d be happy to help with our services. What can I assist you with?",

    # Booking flow
    "booking_question_template": "To help with your booking request, please provide the service you’d like to book, your name, and the best contact number.",
    "booking_details_received": "Thanks — got it! I’ll pass this to the owner to confirm availability and follow up shortly.",

    # Handoff tone variants
    "handoff_normal": "Thanks — got it! I’ll pass this to the owner and they’ll follow up shortly.",
    "handoff_complaint": "I’m sorry about that. I’ll pass this to the owner so they can follow up with you shortly.",

    # Emergency
    "emergency_message": "If this is an emergency, please contact local emergency services immediately.",
}


# =========================
# Regex / Intent helpers
# =========================
EMERGENCY_KEYWORDS = re.compile(r"\b(911|emergency|ambulance|police|fire|danger)\b", re.I)

HOURS_QUERY = re.compile(r"\b(open|close|hours|open today|open now|available today|available now)\b", re.I)
LOCATION_QUERY = re.compile(r"\b(address|location|where are you|directions)\b", re.I)

# Service discovery intent
SERVICE_DISCOVERY = re.compile(
    r"\b(what services|services do you offer|what do you offer|what do you do|services provided|service list)\b",
    re.I,
)

# Booking triggers
BOOKING_TRIGGER = re.compile(r"\b(book|booking|schedule|appointment|appt|reserve)\b", re.I)

# Detect if user likely provided contact info / details
PHONE_LIKE = re.compile(r"\b(\+?\d[\d\-\s]{6,}\d)\b")  # broad phone-like
BOOKING_DETAILS_HINT = re.compile(r"\b(\d{6,}|\d{3}[- ]?\d{3}[- ]?\d{4})\b")

# Complaint / refund / frustration
COMPLAINT_KEYWORDS = re.compile(
    r"\b(refund|return|complain|complaint|bad|terrible|awful|unhappy|angry|upset|frustrated|not happy|issue|problem|charged|scam)\b",
    re.I,
)

# Retail / inventory intent: "do you sell/have/in stock/carry ...", regardless of item
RETAIL_INTENT = re.compile(
    r"\b(do you (sell|have)|do u (sell|have)|in stock|available in stock|carry|selling)\b",
    re.I,
)

# Date/time-only follow-ups (when user responds with "17th", "December 17", "12pm", etc.)
MONTHS = r"(jan|january|feb|february|mar|march|apr|april|may|jun|june|jul|july|aug|august|sep|sept|september|oct|october|nov|november|dec|december)"
DATE_ONLY = re.compile(
    rf"\b({MONTHS}\s*\d{{1,2}}(st|nd|rd|th)?|\d{{1,2}}(st|nd|rd|th)?|tomorrow|today)\b",
    re.I,
)
TIME_ONLY = re.compile(r"\b(\d{1,2}(:\d{2})?\s*(am|pm)|noon|midnight)\b", re.I)


# =========================
# Helpers
# =========================
def now_ts() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def enforce_length(text: str, max_len: int = 240) -> str:
    return clean_text(text)[:max_len]


def looks_like_question(text: str) -> bool:
    t = (text or "").strip()
    if "?" in t:
        return True
    tl = t.lower()
    return tl.startswith(("what", "which", "when", "how", "can you", "could you", "please"))


def is_greeting(text: str) -> bool:
    return clean_text(text).lower() in {"hi", "hello", "hey"}


def is_thanks(text: str) -> bool:
    return clean_text(text).lower() in {"thanks", "thank you", "thx", "ty"}


def likely_booking_details(text: str) -> bool:
    """
    Direct booking-details detector (even if user didn't say 'book'):
    - contains phone-like digits AND
    - contains at least 2 other tokens (name/service)
    Examples:
      "haircut xyz 0123456789" -> True
      "Foot massage john 67890" -> True
    """
    t = clean_text(text)
    if not t:
        return False

    has_phone = bool(BOOKING_DETAILS_HINT.search(t) or PHONE_LIKE.search(t))
    if not has_phone:
        return False

    # Count non-numeric-ish tokens (service/name words)
    tokens = [w for w in re.split(r"[,\s]+", t) if w]
    non_num = [w for w in tokens if not re.fullmatch(r"[\d\-\+\(\)]+", w)]
    return len(non_num) >= 2


def looks_like_date_or_time_only(text: str) -> bool:
    """
    If message is basically just a date/time fragment like:
      "17th", "December 17", "12pm", "tomorrow 11am"
    treat as booking context.
    """
    t = clean_text(text).lower()
    if not t:
        return False

    # If it contains a date/time token and is short-ish, treat it as follow-up info
    has_date = bool(DATE_ONLY.search(t))
    has_time = bool(TIME_ONLY.search(t))
    if not (has_date or has_time):
        return False

    # Keep conservative: if message is short or mostly date/time tokens
    # e.g. "December 17th", "17th", "12pm", "tomorrow 11"
    return len(t.split()) <= 5


def send_alert_if_needed(tags: List[str], needs_handoff: bool, from_number: str, incoming: str, reply_text: str) -> None:
    """
    Email the owner ONLY for important handoffs.
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
        send_handoff_email(
            subject=subject,
            content=body,
            to_email=ALERT_EMAIL_TO,
            from_email=ALERT_EMAIL_FROM,
            api_key=SENDGRID_API_KEY,
        )
    except Exception as e:
        print("SendGrid alert error:", repr(e), flush=True)


def reply_and_log(ts: str, from_number: str, to_number: str, incoming: str, reply_text: str, tags: List[str], needs_handoff: bool):
    reply_text = enforce_length(reply_text)
    log_message(ts, "whatsapp", from_number, to_number, incoming, reply_text, tags, needs_handoff)
    send_alert_if_needed(tags, needs_handoff, from_number, incoming, reply_text)

    resp = MessagingResponse()
    resp.message(reply_text)
    return PlainTextResponse(str(resp), media_type="application/xml")


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

    # Pull last message for minimal follow-up logic
    last = get_last_message_for_sender(from_number)
    last_reply = clean_text((last.get("reply_text") if last else "") or "")
    last_tags_str = (last.get("tags") if last else "") or ""
    last_tags = [t.strip() for t in last_tags_str.split(",") if t.strip()]

    last_was_booking_question = ("booking" in last_tags) and (
        last_reply == BUSINESS_CONTEXT["booking_question_template"] or looks_like_question(last_reply)
    )

    # --------------------------
    # (0) If user sends booking details at ANY time → accept + handoff
    # --------------------------
    if incoming and likely_booking_details(incoming):
        reply_text = BUSINESS_CONTEXT["booking_details_received"]
        return reply_and_log(ts, from_number, to_number, incoming, reply_text, ["booking"], True)

    # --------------------------
    # (A) Follow-up booking details after we asked → accept + handoff
    # (kept for completeness; above rule handles most cases)
    # --------------------------
    if last_was_booking_question and incoming:
        if BOOKING_DETAILS_HINT.search(incoming) or len(incoming.split()) >= 2:
            reply_text = BUSINESS_CONTEXT["booking_details_received"]
            return reply_and_log(ts, from_number, to_number, incoming, reply_text, ["booking"], True)

    # --------------------------
    # (B) Very short messages
    # --------------------------
    if len(incoming) < 3:
        return reply_and_log(ts, from_number, to_number, incoming, BUSINESS_CONTEXT["short_message_clarify"], ["general"], False)

    # --------------------------
    # (C) Thanks / Greeting
    # --------------------------
    if is_thanks(incoming):
        return reply_and_log(ts, from_number, to_number, incoming, BUSINESS_CONTEXT["thanks_reply"], ["general"], False)

    if is_greeting(incoming):
        return reply_and_log(ts, from_number, to_number, incoming, BUSINESS_CONTEXT["greeting"], ["general"], False)

    # --------------------------
    # (D) Emergency / Complaint
    # --------------------------
    if EMERGENCY_KEYWORDS.search(incoming):
        return reply_and_log(ts, from_number, to_number, incoming, BUSINESS_CONTEXT["emergency_message"], ["emergency"], True)

    if COMPLAINT_KEYWORDS.search(incoming):
        return reply_and_log(ts, from_number, to_number, incoming, BUSINESS_CONTEXT["handoff_complaint"], ["complaint"], True)

    # --------------------------
    # (E) Date/time-only follow-ups → treat as booking context
    # --------------------------
    if looks_like_date_or_time_only(incoming):
        return reply_and_log(ts, from_number, to_number, incoming, BUSINESS_CONTEXT["booking_question_template"], ["booking"], True)

    # --------------------------
    # (F) Hours / Location
    # --------------------------
    if HOURS_QUERY.search(incoming):
        return reply_and_log(ts, from_number, to_number, incoming, BUSINESS_CONTEXT["hours_unknown_message"], ["hours"], True)

    if LOCATION_QUERY.search(incoming):
        return reply_and_log(ts, from_number, to_number, incoming, BUSINESS_CONTEXT["location_unknown_message"], ["location"], True)

    # --------------------------
    # (G) Service discovery
    # --------------------------
    if SERVICE_DISCOVERY.search(incoming):
        return reply_and_log(ts, from_number, to_number, incoming, BUSINESS_CONTEXT["service_discovery_message"], ["service_question"], False)

    # --------------------------
    # (H) Booking trigger (deterministic)
    # --------------------------
    if BOOKING_TRIGGER.search(incoming):
        return reply_and_log(ts, from_number, to_number, incoming, BUSINESS_CONTEXT["booking_question_template"], ["booking"], True)

    # --------------------------
    # (I) Retail intent blocker (prevents LLM hallucinations like "we sell trees")
    # If user asks "do you sell/have/carry/in stock" and it's not a booking/service question,
    # respond with retail redirect.
    # --------------------------
    if RETAIL_INTENT.search(incoming):
        return reply_and_log(ts, from_number, to_number, incoming, BUSINESS_CONTEXT["retail_redirect_message"], ["service_question"], False)

    # --------------------------
    # (J) LLM fallback (general)
    # --------------------------
    if client and incoming:
        try:
            completion = client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You reply for a local service business. Be short, professional, and safe. "
                            "Do NOT mention AI, bots, or automation. "
                            "Do NOT say 'Thanks for reaching out'. "
                            "Do NOT claim you sell products or have inventory. "
                            "If asked about products/inventory, redirect to services. "
                            "If you don't know business-specific facts (prices, exact services, hours), ask one simple clarifying question."
                        ),
                    },
                    {"role": "user", "content": incoming},
                ],
                temperature=0.2,
            )
            reply_text = clean_text((completion.choices[0].message.content or "").strip()) or BUSINESS_CONTEXT["greeting"]
            return reply_and_log(ts, from_number, to_number, incoming, reply_text, ["general"], False)
        except Exception as e:
            print("OpenAI error:", repr(e), flush=True)
            return reply_and_log(ts, from_number, to_number, incoming, BUSINESS_CONTEXT["handoff_normal"], ["other"], True)

    return reply_and_log(ts, from_number, to_number, incoming, BUSINESS_CONTEXT["handoff_normal"], ["other"], True)

