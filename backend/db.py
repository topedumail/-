"""SQLite persistence — sessions + feedback.

- sessions: שומר היסטוריית שיחה בין הפעלות שרת
- feedback: שומר לייקים/דיסלייקים על תשובות
"""
from __future__ import annotations

import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Optional

# DATA_DIR מאפשר לאחסן את ה-DB בנפח פרסיסטנטי (Railway Volume וכד').
# כשמשתנה הסביבה לא מוגדר, נופלים חזרה ל-backend/data של הפיתוח הלוקאלי.
_DATA_DIR = Path(os.environ.get("DATA_DIR") or Path(__file__).parent / "data")
DB_PATH = _DATA_DIR / "bot.db"


def _get_conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """יוצר את הטבלאות אם לא קיימות."""
    with _get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id         TEXT PRIMARY KEY,
                history    TEXT NOT NULL DEFAULT '[]',
                updated_at REAL NOT NULL DEFAULT 0
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id     TEXT,
                question       TEXT,
                answer_preview TEXT,
                rating         INTEGER,
                created_at     REAL NOT NULL
            )
        """)
        conn.commit()


def save_session(session_id: str, history: list[dict]) -> None:
    with _get_conn() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO sessions (id, history, updated_at) VALUES (?, ?, ?)",
            (session_id, json.dumps(history, ensure_ascii=False), time.time()),
        )
        conn.commit()


def load_session(session_id: str) -> Optional[list[dict]]:
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT history FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
        if row:
            try:
                return json.loads(row["history"])
            except Exception:
                pass
    return None


def delete_session(session_id: str) -> None:
    with _get_conn() as conn:
        conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        conn.commit()


def save_feedback(
    session_id: str, question: str, answer_preview: str, rating: int
) -> None:
    with _get_conn() as conn:
        conn.execute(
            """INSERT INTO feedback
               (session_id, question, answer_preview, rating, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            (session_id, question[:600], answer_preview[:400], rating, time.time()),
        )
        conn.commit()
