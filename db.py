import sqlite3
from typing import List, Dict, Any, Optional

DB_PATH = "app.db"


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    conn = get_conn()
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                channel TEXT NOT NULL,
                from_number TEXT NOT NULL,
                to_number TEXT,
                inbound_text TEXT NOT NULL,
                reply_text TEXT NOT NULL,
                tags TEXT NOT NULL,
                needs_handoff INTEGER NOT NULL
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


def log_message(
    ts: str,
    channel: str,
    from_number: str,
    to_number: Optional[str],
    inbound_text: str,
    reply_text: str,
    tags: List[str],
    needs_handoff: bool,
) -> None:
    conn = get_conn()
    try:
        conn.execute(
            """
            INSERT INTO messages (ts, channel, from_number, to_number, inbound_text, reply_text, tags, needs_handoff)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ts,
                channel,
                from_number,
                to_number,
                inbound_text,
                reply_text,
                ",".join(tags),
                1 if needs_handoff else 0,
            ),
        )
        conn.commit()
    finally:
        conn.close()


def get_recent_messages(limit: int = 100) -> List[Dict[str, Any]]:
    conn = get_conn()
    try:
        cur = conn.execute(
            """
            SELECT id, ts, channel, from_number, to_number, inbound_text, reply_text, tags, needs_handoff
            FROM messages
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        )
        rows = cur.fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_last_message_for_sender(from_number: str) -> Optional[Dict[str, Any]]:
    conn = get_conn()
    try:
        cur = conn.execute(
            """
            SELECT id, ts, channel, from_number, to_number, inbound_text, reply_text, tags, needs_handoff
            FROM messages
            WHERE from_number = ?
            ORDER BY id DESC
            LIMIT 1
            """,
            (from_number,),
        )
        row = cur.fetchone()
        return dict(row) if row else None
    finally:
        conn.close()

