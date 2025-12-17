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

    # Retail / irrelevant product questions (eggs/milk/toys/gems)
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

# Retail-ish keywords (MVP heuristic; can expand later)
RETAIL_KEYWORDS = re.compile(r"\b(milk|eggs|bread|toys?|gems?|diaper|cigarettes|soda|chips)\b", re.I)

# Service discovery intent
SERVICE_DISCOVERY = re.compile(r"\b(what services|services do you offer|what do you offer|what do you do|services provided|service list)\b", re.I)

# Booking triggers
BOOKING_TRIGGER = re.compile(r"\b(book|booking|schedule|appointment|appt|reserve)\b", re.I)

# Detect if user likely provided contact info / details
BOOKING_DETAILS_HINT = re.compile(r"\b(\d{6,}|\d{3}[- ]?\d{3}[- ]?\d{4})\b")

# Complaint / refund / frustration
COMPLAINT_KEYWORDS = re.compile(
    r"\b(refund|return|complain|complaint|bad|terrible|awful|unhappy|angry|upset|frustrated|not happy|issue|problem|charged|scam)\b",
    re.I,
)


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
    reply_text = BUSINESS_CONTEXT["handoff_normal"]
    needs_handoff = True
    tags: List[str] = ["general"]

    # Pull last message for simple follow-up logic (booking details)
    last = get_last_message_for_sender(from_number)
    last_reply = clean_text((last.get("reply_text") if last else "") or "")
    last_tags_str = (last.get("tags") if last else "") or ""
    last_tags = [t.strip() for t in last_tags_str.split(",") if t.strip()]

    last_was_booking_question = ("booking" in last_tags) and looks_like_question(last_reply)

    # --------------------------
    # (A) Follow-up booking details
    # If we asked for booking details previously and user now sends likely details, acknowledge & handoff.
    # --------------------------
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

    # --------------------------
    # (B) Very short messages
    # --------------------------
    if len(incoming) < 3:
        reply_text = BUSINESS_CONTEXT["short_message_clarify"]
        tags = ["general"]
        needs_handoff = False

    # --------------------------
    # (C) Thanks
    # --------------------------
    elif incoming.lower() in {"thanks", "thank you", "thx", "ty"}:
        reply_text = BUSINESS_CONTEXT["thanks_reply"]
        tags = ["general"]
        needs_handoff = False

    # --------------------------
    # (D) Greeting
    # --------------------------
    elif incoming.lower() in {"hi", "hello", "hey"}:
        reply_text = BUSINESS_CONTEXT["greeting"]
        tags = ["general"]
        needs_handoff = False

    # --------------------------
    # (E) Emergency
    # --------------------------
    elif EMERGENCY_KEYWORDS.search(incoming):
        reply_text = BUSINESS_CONTEXT["emergency_message"]
        tags = ["emergency"]
        needs_handoff = True

    # --------------------------
    # (F) Complaint / refund / frustration → different handoff tone
    # --------------------------
    elif COMPLAINT_KEYWORDS.search(incoming):
        reply_text = BUSINESS_CONTEXT["handoff_complaint"]
        tags = ["complaint"]
        needs_handoff = True

    # --------------------------
    # (G) Retail / irrelevant product questions → redirect to our services
    # --------------------------
    elif RETAIL_KEYWORDS.search(incoming):
        reply_text = BUSINESS_CONTEXT["retail_redirect_message"]
        tags = ["service_question"]
        needs_handoff = False

    # --------------------------
    # (H) Service discovery → do NOT redirect as retail; ask what they need (MVP generic)
    # --------------------------
    elif SERVICE_DISCOVERY.search(incoming):
        reply_text = BUSINESS_CONTEXT["service_discovery_message"]
        tags = ["service_question"]
        needs_handoff = False

    # --------------------------
    # (I) Hours / availability
    # --------------------------
    elif HOURS_QUERY.search(incoming):
        reply_text = BUSINESS_CONTEXT["hours_unknown_message"]
        tags = ["hours"]
        needs_handoff = True

    # --------------------------
    # (J) Location
    # --------------------------
    elif LOCATION_QUERY.search(incoming):
        reply_text = BUSINESS_CONTEXT["location_unknown_message"]
        tags = ["location"]
        needs_handoff = True

    # --------------------------
    # (K) Booking trigger (deterministic)
    # --------------------------
    elif BOOKING_TRIGGER.search(incoming):
        reply_text = BUSINESS_CONTEXT["booking_question_template"]
        tags = ["booking"]
        needs_handoff = True

    # --------------------------
    # (L) LLM fallback (only for general queries)
    # --------------------------
    else:
        if client and incoming:
            try:
                completion = client.chat.completions.create(
                    model="gpt-4.1-mini",
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You reply for a local service business. "
                                "Be short, professional, and safe. "
                                "Do not mention AI, bots, or automation. "
                                "If you don't know business-specific facts (prices, exact services, hours), ask a simple clarifying question or hand off politely."
                            ),
                        },
                        {"role": "user", "content": incoming},
                    ],
                    temperature=0.2,
                )
                reply_text = clean_text((completion.choices[0].message.content or "").strip())
                tags = ["general"]
                needs_handoff = False
            except Exception as e:
                print("OpenAI error:", repr(e), flush=True)
                reply_text = BUSINESS_CONTEXT["handoff_normal"]
                tags = ["other"]
                needs_handoff = True
        else:
            reply_text = BUSINESS_CONTEXT["handoff_normal"]
            tags = ["other"]
            needs_handoff = True

    reply_text = enforce_length(reply_text)

    # Log + alert
    log_message(ts, "whatsapp", from_number, to_number, incoming, reply_text, tags, needs_handoff)
    send_alert_if_needed(tags, needs_handoff, from_number, incoming, reply_text)

    # Respond
    resp = MessagingResponse()
    resp.message(reply_text)
    return PlainTextResponse(str(resp), media_type="application/xml")

