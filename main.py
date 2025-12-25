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

# availability intent (this fixes "Do you have availability for 12PM?")
AVAILABILITY_INTENT = re.compile(r"\b(availability|available)\b", re.I)

# Date/time tokens
MONTHS = r"(jan|january|feb|february|mar|march|apr|april|may|jun|june|jul|july|aug|august|sep|sept|september|oct|october|nov|november|dec|december)"
DATE_TOKEN = re.compile(rf"\b({MONTHS}\s*\d{{1,2}}(st|nd|rd|th)?|\d{{1,2}}(st|nd|rd|th)?|tomorrow|today)\b", re.I)
TIME_TOKEN = re.compile(r"\b(\d{1,2}(:\d{2})?\s*(am|pm)|noon|midnight)\b", re.I)

COMPLAINT_KEYWORDS = re.compile(
    r"\b(refund|return|complain|complaint|bad|terrible|awful|unhappy|angry|upset|frustrated|not happy|issue|problem|charged|scam)\b",
    re.I,
)

# Retail intent blocker (prevent hallucinations like "we sell trees")
RETAIL_INTENT = re.compile(
    r"\b(do you (sell|have)|do u (sell|have)|in stock|available in stock|carry|selling)\b",
    re.I,
)

# Booking details detector:
# - A phone-like number: 7+ digits OR common phone formatting
PHONE_LIKE_STRICT = re.compile(r"\b(\+?\d[\d\-\s]{6,}\d)\b")  # 7+ digits overall
DIGITS_7PLUS = re.compile(r"\b\d{7,}\b")  # safer for phone than 5-digit ZIP
NAME_PLUS_NUMBER = re.compile(r"\b([a-zA-Z]{2,})\b.*\b\d{7,}\b")  # word + 7+ digits


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
    """
    Accept booking details when message includes a likely phone number (7+ digits)
    and at least one word (name/service). Avoid treating 5-digit ZIP codes as phone.
    """
    t = clean_text(text)
    if not t:
        return False

    has_phone = bool(DIGITS_7PLUS.search(t) or PHONE_LIKE_STRICT.search(t))
    if not has_phone:
        return False

    # require at least one alpha token
    has_word = bool(re.search(r"[A-Za-z]{2,}", t))
    return has_word


def looks_like_booking_partial(text: str) -> bool:
    """
    Booking-intent message where they gave service/name and *some* short number token (like ZIP),
    but NOT a real phone number. Example: "Foot massage john 67890"
    We treat this as booking intent and ask the standard booking template.
    """
    t = clean_text(text)
    if not t:
        return False

    # Need at least two alpha tokens (service + name-ish)
    alpha_tokens = re.findall(r"[A-Za-z]{2,}", t)
    if len(alpha_tokens) < 2:
        return False

    # Has a 3–5 digit token (likely ZIP / short code)
    has_short_number = bool(re.search(r"\b\d{3,5}\b", t))

    # Must NOT already have a phone-like 7+ digit number
    has_phone = bool(DIGITS_7PLUS.search(t) or PHONE_LIKE_STRICT.search(t))

    return has_short_number and (not has_phone)


def send_alert_if_needed(tags: List[str], needs_handoff: bool, from_number: str, incoming: str, reply_text: str) -> None:
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

    # Persist inbound source in tags (no schema change)
    source = detect_source(incoming)
    src_tag = f"src:{source}"
    if src_tag not in tags:
        tags = tags + [src_tag]

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

    last = get_last_message_for_sender(from_number)
    last_reply = clean_text((last.get("reply_text") if last else "") or "")
    last_tags_str = (last.get("tags") if last else "") or ""
    last_tags = [t.strip() for t in last_tags_str.split(",") if t.strip()]
    last_was_booking_question = ("booking" in last_tags) and (last_reply == BUSINESS_CONTEXT["booking_question_template"])

    # 0) Direct booking details any time (with 7+ digit phone)
    if incoming and looks_like_booking_details(incoming):
        return reply_and_log(ts, from_number, to_number, incoming, BUSINESS_CONTEXT["booking_details_received"], ["booking"], True)

    # 0b) Booking partial (service+name + short number like ZIP) -> treat as booking intent
    if incoming and looks_like_booking_partial(incoming):
        return reply_and_log(ts, from_number, to_number, incoming, BUSINESS_CONTEXT["booking_question_template"], ["booking"], True)

    # 1) Very short messages
    if len(incoming) < 3:
        return reply_and_log(ts, from_number, to_number, incoming, BUSINESS_CONTEXT["short_message_clarify"], ["general"], False)

    # 2) Thanks / greeting
    if is_thanks(incoming):
        return reply_and_log(ts, from_number, to_number, incoming, BUSINESS_CONTEXT["thanks_reply"], ["general"], False)

    if is_greeting(incoming):
        return reply_and_log(ts, from_number, to_number, incoming, BUSINESS_CONTEXT["greeting"], ["general"], False)

    # 3) Emergency / complaint
    if EMERGENCY_KEYWORDS.search(incoming):
        return reply_and_log(ts, from_number, to_number, incoming, BUSINESS_CONTEXT["emergency_message"], ["emergency"], True)

    if COMPLAINT_KEYWORDS.search(incoming):
        return reply_and_log(ts, from_number, to_number, incoming, BUSINESS_CONTEXT["handoff_complaint"], ["complaint"], True)

    # 4) Service discovery MUST win over booking follow-up
    if SERVICE_DISCOVERY.search(incoming):
        return reply_and_log(ts, from_number, to_number, incoming, BUSINESS_CONTEXT["service_discovery_message"], ["service_question"], False)

    # 5) Availability + time/date => booking intent (fixes "availability for 12PM")
    if AVAILABILITY_INTENT.search(incoming) and (TIME_TOKEN.search(incoming) or DATE_TOKEN.search(incoming)):
        return reply_and_log(ts, from_number, to_number, incoming, BUSINESS_CONTEXT["booking_question_template"], ["booking"], True)

    # 6) Hours / Location
    if HOURS_QUERY.search(incoming):
        return reply_and_log(ts, from_number, to_number, incoming, BUSINESS_CONTEXT["hours_unknown_message"], ["hours"], True)

    if LOCATION_QUERY.search(incoming):
        return reply_and_log(ts, from_number, to_number, incoming, BUSINESS_CONTEXT["location_unknown_message"], ["location"], True)

    # 7) Booking trigger
    if BOOKING_TRIGGER.search(incoming):
        return reply_and_log(ts, from_number, to_number, incoming, BUSINESS_CONTEXT["booking_question_template"], ["booking"], True)

    # 8) Follow-up after booking question: ONLY accept if message contains 7+ digit phone
    # (fixes "what services do you offer?" being mis-classified)
    if last_was_booking_question and incoming:
        if looks_like_booking_details(incoming):
            return reply_and_log(ts, from_number, to_number, incoming, BUSINESS_CONTEXT["booking_details_received"], ["booking"], True)
        # otherwise, re-ask booking template (don't guess)
        return reply_and_log(ts, from_number, to_number, incoming, BUSINESS_CONTEXT["booking_question_template"], ["booking"], True)

    # 9) Retail intent blocker (prevents hallucinations like "we sell trees")
    if RETAIL_INTENT.search(incoming):
        return reply_and_log(ts, from_number, to_number, incoming, BUSINESS_CONTEXT["retail_redirect_message"], ["service_question"], False)

    # 10) LLM fallback (general)
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

