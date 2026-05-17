"""Query the educational pedagogy bot deployed on Render.

Usage:
    python query.py "השאלה שלי"

Output: prints the answer followed by the cited sources.
"""
from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request

BOT_URL = "https://tofes.onrender.com/api/chat/stream"
STATUS_URL = "https://tofes.onrender.com/api/status"
TIMEOUT_SECONDS = 180   # זמן מקסימלי לבקשת השיחה
WAKEUP_MAX_WAIT = 180   # זמן מקסימלי להמתנה לאינדקס שיהיה מוכן
POLL_INTERVAL = 5       # בודק כל 5 שניות


def _wait_for_ready() -> None:
    """ממתין עד שהאינדקס מוכן (Render Free Tier sleep + index build)."""
    deadline = time.time() + WAKEUP_MAX_WAIT
    last_error: str = ""
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(STATUS_URL, timeout=30) as resp:
                if resp.status == 200:
                    return  # האינדקס מוכן
        except urllib.error.HTTPError as e:
            if e.code == 503:
                last_error = "המאגר נטען..."
            else:
                last_error = f"HTTP {e.code}"
        except urllib.error.URLError as e:
            last_error = f"רשת: {e.reason}"
        except Exception as e:
            last_error = str(e)
        time.sleep(POLL_INTERVAL)
    raise SystemExit(
        f"השרת לא היה מוכן בתוך {WAKEUP_MAX_WAIT} שניות. "
        f"שגיאה אחרונה: {last_error}"
    )


def ask_bot(question: str) -> tuple[str, list[dict]]:
    """שולח שאלה לבוט ומחזיר (תשובה, רשימת מקורות)."""
    _wait_for_ready()

    payload = json.dumps({"question": question}, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        BOT_URL,
        data=payload,
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )

    sources: list[dict] = []
    answer_chunks: list[str] = []

    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_SECONDS) as resp:
            buffer = ""
            while True:
                chunk = resp.read(4096)
                if not chunk:
                    break
                buffer += chunk.decode("utf-8", errors="replace")

                # SSE: events מופרדים ב-\n\n
                while "\n\n" in buffer:
                    event_text, buffer = buffer.split("\n\n", 1)
                    for line in event_text.split("\n"):
                        if not line.startswith("data: "):
                            continue
                        try:
                            event = json.loads(line[6:])
                        except json.JSONDecodeError:
                            continue

                        event_type = event.get("type")
                        if event_type == "sources":
                            sources = event.get("matches", [])
                        elif event_type == "delta":
                            answer_chunks.append(event.get("text", ""))
                        elif event_type == "done":
                            return "".join(answer_chunks), sources
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise SystemExit(f"שגיאת HTTP {e.code}: {body}")
    except urllib.error.URLError as e:
        raise SystemExit(f"שגיאת רשת: {e.reason}")

    return "".join(answer_chunks), sources


def main() -> None:
    if len(sys.argv) < 2:
        print("שימוש: python query.py \"השאלה שלך\"", file=sys.stderr)
        sys.exit(2)

    question = " ".join(sys.argv[1:]).strip()
    if not question:
        print("שאלה ריקה", file=sys.stderr)
        sys.exit(2)

    # הגדרת stdout כ-UTF-8 (חשוב ל-Windows Console)
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    answer, sources = ask_bot(question)

    print(answer.strip())
    print()
    if sources:
        print("─" * 60)
        print("מקורות שאוחזרו:")
        seen = set()
        for s in sources:
            label = s.get("source", "?")
            section = s.get("section") or ""
            key = (label, section)
            if key in seen:
                continue
            seen.add(key)
            section_str = f" § {section}" if section else ""
            print(f"  • {label}{section_str}")


if __name__ == "__main__":
    main()
