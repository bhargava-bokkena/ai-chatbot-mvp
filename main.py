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


# =========================
# Setup
# =========================
load_dotenv()
init_db()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

DASH_TOKEN = os.getenv("DASH_TOKEN", "").strip()

app = FastAPI()


# =========================
# Business Context (MVP)
# =========================
BUSINESS_CONTEXT = {
    "business_name": "This business",
    "business_type": "Local service business",
    "location": "New York, NY",
    "tone": "professional",

    "hours": "UNKNOWN",
    "services": "UNKNOWN",

    "booking_method": "text",
    "phone": "UNKNOWN",
    "website": "UNKNOWN",

    "handoff_message": "Thanks for reaching out — I’m going to pass this to the owner and they’ll follow up shortly.",
    "hours_unknown_message": "I don’t have the hours handy right now, but I can check with the owner and get back to you shortly.",
    "location_unknown_message": "I don’t have the address handy right now, but I can check with the owner and follow up shortly.",

    # Inventory fix (consistent phrasing)
    "inventory_unknown_message": "We don’t carry retail products like that. We primarily offer services — would you like help with any of our services?",

    "emergency_message": "If this is an emergency or someone is in danger, please call 911 right now. If you’re safe, share a brief summary and I’ll pass it to the owner.",
    "short_message_clarify": "Sorry — could you share a bit more detail so I can help?",

    # Booking: make the question consistent (deterministic)
    "booking_question_template": "To help with your booking request, what service is it for, what’s your name, and the best contact number?",
    "booking_details_received": "Thanks — got it. I’ll pass this to the owner to confirm availability and follow up shortly.",
}


# =========================
# Regex shortcuts
# =========================
EMERGENCY_KEYWORDS = re.compile(
    r"\b(911|emergency|ambulance|police|fire|bleeding|unconscious|overdose|suicide|kill myself|harm myself|gun|shoot|stabbing|attack|danger)\b",
    re.IGNORECASE
)

HOURS_QUERY = re.compile(
    r"\b(hours|open|close|closing|opening|are you open|are you closed|open today|open now)\b",
    re.IGNORECASE
)

LOCATION_QUERY = re.compile(
    r"\b(address|location|located|where are you|directions|near you)\b",
    re.IGNORECASE
)

INVENTORY_KEYWORDS = re.compile(
    r"\b(milk|eggs|bread|water|soda|juice|coffee|tea|in stock|available|carry|sell|stock|toys|gems)\b",
    re.IGNORECASE
)

BOOKING_DETAILS_HINT = re.compile(r"\b(\d{6,}|\d{3}[-\s]?\d{3}[-\s]?\d{4})\b")


# =========================
# LLM Prompt
# =========================
SYSTEM_PROMPT = f"""
You reply to customer messages on behalf of a local business.

BUSINESS CONTEXT (source of truth):
- Name: {BUSINESS_CONTEXT["business_name"]}
- Type: {BUSINESS_CONTEXT["business_type"]}
- Location: {BUSINESS_CONTEXT["location"]}
- Hours: {BUSINESS_CONTEXT["hours"]}
- Services: {BUSINESS_CONTEXT["services"]}
- Booking method: {BUSINESS_CONTEXT["booking_method"]}
- Phone: {BUSINESS_CONTEXT["phone"]}
- Website: {BUSINESS_CONTEXT["website"]}
- Tone: {BUSINESS_CONTEXT["tone"]}

STYLE CONSTRAINTS:
- 1–2 sentences, max ~240 characters.
- One message only (no paragraphs/bullets).
- No emojis unless tone is casual.

SAFETY RULES:
- Do NOT confirm hours, pricing, availability, bookings, addresses, or policies unless explicitly provided in context.
- Hours question: do NOT ask what day it is. If hours unknown, offer to check with owner (needs_handoff=true).
- Pricing unknown: ask ONE clarifying question; do NOT handoff unless necessary.
- Booking: do NOT confirm. Ask for (1) service, (2) name, (3) best contact. Set needs_handoff=true.
- Complaints/refunds/legal threats: set needs_handoff=true.
- Emergency: advise contacting local emergency services and set needs_handoff=true.
- If service-based and asked about physical products/inventory, say you don’t carry retail products and offer help with your services instead.
- Never mention you are an AI.

OUTPUT JSON ONLY:
{{
  "reply": "string",
  "needs_handoff": true/false,
  "tags": ["pricing"|"hours"|"booking"|"location"|"service_question"|"general"|"complaint"|"emergency"|"other"]
}}
""".strip()


ALLOWED_TAGS = {
    "pricing", "hours", "booking", "location",
    "service_question", "general", "complaint",
    "emergency", "other"
}

ALWAYS_HANDOFF_TAGS = {"complaint", "emergency"}


# =========================
# Helper funcs
# =========================
def now_ts() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def safe_parse_json(text: str) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(text)
    except Exception:
        return None


def normalize_tags(tags: Any) -> List[str]:
    if not isinstance(tags, list):
        return ["other"]
    cleaned = []
    for t in tags:
        if isinstance(t, str):
            tt = t.strip().lower()
            if tt in ALLOWED_TAGS:
                cleaned.append(tt)
    return cleaned or ["other"]


def is_unknown(value: str) -> bool:
    return (value or "").strip().upper() == "UNKNOWN"


def looks_like_info_collection(reply_text: str) -> bool:
    t = reply_text.lower().strip()
    return (
        "?" in reply_text
        or t.startswith(("what", "which", "when", "where", "who", "how"))
        or "can you" in t
        or "could you" in t
        or "please" in t
    )


def enforce_style(reply_text: str) -> str:
    msg = re.sub(r"\s+", " ", reply_text).strip()
    if len(msg) > 240:
        msg = msg[:237].rstrip() + "..."
    return msg


def detect_channel(from_number: str) -> str:
    return "whatsapp" if from_number.lower().startswith("whatsapp:") else "sms"


def finalize_reply(reply_text: str, needs_handoff: bool, tags: List[str]) -> str:
    if "emergency" in tags:
        return BUSINESS_CONTEXT["emergency_message"]

    if "hours" in tags and is_unknown(BUSINESS_CONTEXT["hours"]):
        return BUSINESS_CONTEXT["hours_unknown_message"]

    if "location" in tags and is_unknown(BUSINESS_CONTEXT["location"]):
        return BUSINESS_CONTEXT["location_unknown_message"]

    if "complaint" in tags:
        return BUSINESS_CONTEXT["handoff_message"]

    # Booking: if collecting details, do NOT append handoff in same message
    if "booking" in tags and needs_handoff and looks_like_info_collection(reply_text):
        return reply_text.strip()

    if not needs_handoff:
        return reply_text.strip()

    if looks_like_info_collection(reply_text):
        return reply_text.strip()

    return BUSINESS_CONTEXT["handoff_message"]


# =========================
# Routes
# =========================
@app.get("/health")
def health():
    return {"ok": True}

@app.get("/")
def root():
    return {"ok": True, "service": "ai-reply-assistant"}

@app.get("/logs")
def logs(token: str = ""):
    if not DASH_TOKEN:
        raise HTTPException(status_code=404, detail="Not Found")
    if token != DASH_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")

    rows = get_recent_messages(limit=200)
    html = [
        "<html><head><meta charset='utf-8'><title>Message Logs</title></head><body>",
        "<h2>Recent Messages</h2>",
        "<table border='1' cellpadding='6' cellspacing='0'>",
        "<tr><th>ID</th><th>TS</th><th>Channel</th><th>From</th><th>Inbound</th><th>Reply</th><th>Tags</th><th>Handoff</th></tr>",
    ]
    for r in rows:
        html.append(
            "<tr>"
            f"<td>{r['id']}</td>"
            f"<td>{r['ts']}</td>"
            f"<td>{r['channel']}</td>"
            f"<td>{r['from_number']}</td>"
            f"<td>{(r['inbound_text'] or '').replace('<','&lt;')}</td>"
            f"<td>{(r['reply_text'] or '').replace('<','&lt;')}</td>"
            f"<td>{r['tags']}</td>"
            f"<td>{'yes' if r['needs_handoff'] else 'no'}</td>"
            "</tr>"
        )
    html.append("</table></body></html>")
    return HTMLResponse("".join(html))


@app.post("/webhook/inbound")
async def inbound_message(request: Request):
    form = await request.form()
    incoming = (form.get("Body") or "").strip()
    from_number = (form.get("From") or "").strip()
    to_number = (form.get("To") or "").strip()

    channel = detect_channel(from_number)
    ts = now_ts()

    # Default
    reply_text = BUSINESS_CONTEXT["handoff_message"]
    needs_handoff = True
    tags: List[str] = ["other"]

    # --- Minimal “conversation memory” for booking follow-up ---
    last = get_last_message_for_sender(from_number)
    last_tags_str = (last.get("tags", "") if last else "") or ""
    last_reply = (last.get("reply_text", "") if last else "") or ""
    last_tags = [t for t in last_tags_str.split(",") if t.strip()]

    last_was_booking_question = ("booking" in last_tags) and looks_like_info_collection(last_reply)

    if last_was_booking_question and ("booking" not in incoming.lower()):
        if BOOKING_DETAILS_HINT.search(incoming) or len(incoming.split()) >= 2:
            reply_text = BUSINESS_CONTEXT["booking_details_received"]
            needs_handoff = True
            tags = ["booking"]

            reply_text = enforce_style(reply_text)
            log_message(ts, channel, from_number, to_number, incoming, reply_text, tags, needs_handoff)

            resp = MessagingResponse()
            resp.message(reply_text)
            return PlainTextResponse(str(resp), media_type="application/xml")

    # 0) Very short messages
    if len(incoming) <= 2:
        reply_text = BUSINESS_CONTEXT["short_message_clarify"]
        needs_handoff = False
        tags = ["general"]

        reply_text = enforce_style(reply_text)
        log_message(ts, channel, from_number, to_number, incoming, reply_text, tags, needs_handoff)

        resp = MessagingResponse()
        resp.message(reply_text)
        return PlainTextResponse(str(resp), media_type="application/xml")

    # 1) Emergency
    if EMERGENCY_KEYWORDS.search(incoming):
        reply_text = BUSINESS_CONTEXT["emergency_message"]
        needs_handoff = True
        tags = ["emergency"]

        reply_text = enforce_style(reply_text)
        log_message(ts, channel, from_number, to_number, incoming, reply_text, tags, needs_handoff)

        resp = MessagingResponse()
        resp.message(reply_text)
        return PlainTextResponse(str(resp), media_type="application/xml")

    # 2) Inventory/product fix
    if INVENTORY_KEYWORDS.search(incoming) and (
        "service" in BUSINESS_CONTEXT["business_type"].lower() or is_unknown(BUSINESS_CONTEXT["services"])
    ):
        reply_text = BUSINESS_CONTEXT["inventory_unknown_message"]
        needs_handoff = False
        tags = ["service_question"]

        reply_text = enforce_style(reply_text)
        log_message(ts, channel, from_number, to_number, incoming, reply_text, tags, needs_handoff)

        resp = MessagingResponse()
        resp.message(reply_text)
        return PlainTextResponse(str(resp), media_type="application/xml")

    # 3) Hours shortcut
    if HOURS_QUERY.search(incoming) and is_unknown(BUSINESS_CONTEXT["hours"]):
        reply_text = BUSINESS_CONTEXT["hours_unknown_message"]
        needs_handoff = True
        tags = ["hours"]

        reply_text = enforce_style(reply_text)
        log_message(ts, channel, from_number, to_number, incoming, reply_text, tags, needs_handoff)

        resp = MessagingResponse()
        resp.message(reply_text)
        return PlainTextResponse(str(resp), media_type="application/xml")

    # 4) Location shortcut
    if LOCATION_QUERY.search(incoming) and is_unknown(BUSINESS_CONTEXT["location"]):
        reply_text = BUSINESS_CONTEXT["location_unknown_message"]
        needs_handoff = True
        tags = ["location"]

        reply_text = enforce_style(reply_text)
        log_message(ts, channel, from_number, to_number, incoming, reply_text, tags, needs_handoff)

        resp = MessagingResponse()
        resp.message(reply_text)
        return PlainTextResponse(str(resp), media_type="application/xml")

    # 5) LLM
    if client and incoming:
        try:
            completion = client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Customer ({from_number}) message: {incoming}"},
                ],
                temperature=0.2,
            )

            content = (completion.choices[0].message.content or "").strip()
            data = safe_parse_json(content)

            if isinstance(data, dict) and isinstance(data.get("reply"), str):
                raw_reply = data["reply"].strip()
                raw_tags = normalize_tags(data.get("tags", ["other"]))
                raw_needs_handoff = bool(data.get("needs_handoff", False))

                needs_handoff = raw_needs_handoff or any(t in ALWAYS_HANDOFF_TAGS for t in raw_tags)
                tags = raw_tags

                reply_text = raw_reply if raw_reply else BUSINESS_CONTEXT["handoff_message"]
                reply_text = finalize_reply(reply_text, needs_handoff, tags)

                # ✅ CONSISTENCY OVERRIDE:
                # If this is a booking info-collection message, force our exact template wording.
                if "booking" in tags and needs_handoff and looks_like_info_collection(reply_text):
                    reply_text = BUSINESS_CONTEXT["booking_question_template"]

            else:
                reply_text = BUSINESS_CONTEXT["handoff_message"]
                needs_handoff = True
                tags = ["other"]

        except Exception as e:
            print("OpenAI error:", repr(e))
            reply_text = BUSINESS_CONTEXT["handoff_message"]
            needs_handoff = True
            tags = ["other"]

    reply_text = enforce_style(reply_text)

    log_message(ts, channel, from_number, to_number, incoming, reply_text, tags, needs_handoff)

    resp = MessagingResponse()
    resp.message(reply_text)
    return PlainTextResponse(str(resp), media_type="application/xml")

