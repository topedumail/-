"""FastAPI app — בוט מענה אישי עם hybrid search, conversation memory, ו-streaming."""
from __future__ import annotations

import json
import os
import queue
import shutil
import threading
import traceback
import uuid
from pathlib import Path

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from openai import OpenAI
from pydantic import BaseModel

import db
from llm import ChatSession, MAX_HISTORY_TURNS, stream_answer, generate_followups
from rag import HybridIndex

# --- הגדרות ---
BACKEND_DIR = Path(__file__).parent
PROJECT_DIR = BACKEND_DIR.parent
load_dotenv(PROJECT_DIR / ".env")

# DATA_DIR מאפשר לאחסן cache + db בנפח פרסיסטנטי (Railway Volume וכד').
# ברירת מחדל — תיקיית data שבתוך backend (מתאים לפיתוח לוקאלי).
DATA_DIR = Path(os.environ.get("DATA_DIR") or BACKEND_DIR / "data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# מסמכי הידע תמיד נטענים מהריפו (חלק מהקוד, לא נתון משתמש)
KNOWLEDGE_DIR = BACKEND_DIR / "data" / "knowledge"
# Cache + DB מוצאים מהריפו אל DATA_DIR — שורדים deploys כשעל volume
CACHE_PATH = DATA_DIR / ".embed_cache.pkl"
FRONTEND_DIR = PROJECT_DIR / "frontend"

API_KEY = os.environ.get("OPENAI_API_KEY")
if not API_KEY:
    print("[main] אזהרה: לא הוגדר OPENAI_API_KEY. הבוט לא יענה עד שיוגדר.")

# Timeouts מותאמים למודל חשיבה (gpt-5.x):
#  - connect: 15ש' להקמת החיבור
#  - read: 120ש' בין chunks — מכסה זמן "חשיבה" ארוך לפני/בין טוקנים
#  - write/pool: 15ש'
# max_retries=2 → ניסיון חוזר אוטומטי (backoff) על 429/5xx זמניים.
_OPENAI_TIMEOUT = httpx.Timeout(120.0, connect=15.0, write=15.0, pool=15.0)
client = (
    OpenAI(api_key=API_KEY, timeout=_OPENAI_TIMEOUT, max_retries=2)
    if API_KEY
    else None
)
index = HybridIndex(client=client, cache_path=CACHE_PATH) if client else None

# Sessions בזיכרון. בפרודקשן צריך Redis או DB.
_sessions: dict[str, ChatSession] = {}

# סטטוס בניית האינדקס — כדי שהשרת יוכל לדווח אם האינדקס מוכן.
_index_state = {"ready": False, "building": False, "error": None}


def _get_session(session_id: str | None) -> tuple[str, ChatSession]:
    if session_id:
        # קודם מה-cache בזיכרון
        if session_id in _sessions:
            return session_id, _sessions[session_id]
        # אחר כך מ-SQLite
        history = db.load_session(session_id)
        if history is not None:
            session = ChatSession.from_dict_list(history)
            _sessions[session_id] = session
            return session_id, session
    new_id = str(uuid.uuid4())
    _sessions[new_id] = ChatSession()
    return new_id, _sessions[new_id]


# --- מודלים ---


class ChatRequest(BaseModel):
    question: str
    session_id: str | None = None


class FeedbackRequest(BaseModel):
    session_id: str | None = None
    question: str
    answer_preview: str
    rating: int  # 1 = 👍, -1 = 👎


class IndexStatus(BaseModel):
    documents: int
    chunks: int
    sources: list[str]

ALLOWED_EXTENSIONS = {".txt", ".md", ".markdown", ".docx", ".pdf", ".pptx"}


# --- אפליקציה ---
app = FastAPI(title="בוט אגף בכיר חינוך ילדים ונוער בסיכון", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _build_index_in_background() -> None:
    """בונה את האינדקס ברקע — לא חוסם את עליית השרת."""
    _index_state["building"] = True
    try:
        KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)
        summary = index.build(KNOWLEDGE_DIR)
        print(
            f"[main] נטענו {summary['documents']} מסמכים, "
            f"{summary['chunks']} chunks ({summary['new']} חדשים, {summary['cached']} מ-cache).",
            flush=True,
        )
        _index_state["ready"] = True
    except Exception as e:
        _index_state["error"] = str(e)
        print(f"[main] שגיאה בבניית האינדקס: {e}", flush=True)
        traceback.print_exc()
    finally:
        _index_state["building"] = False


@app.on_event("startup")
def on_startup() -> None:
    """מפעיל את בניית האינדקס ב-thread נפרד — השרת מאזין על הפורט מיד."""
    db.init_db()
    print("[main] SQLite אותחל.", flush=True)
    if index is None:
        print("[main] אין מפתח OpenAI — האינדקס לא ייבנה.", flush=True)
        return
    threading.Thread(target=_build_index_in_background, daemon=True).start()
    print("[main] השרת עלה. בניית האינדקס רצה ברקע.", flush=True)


@app.get("/")
def root() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/api/status", response_model=IndexStatus)
def status() -> IndexStatus:
    if index is None:
        raise HTTPException(503, "OPENAI_API_KEY לא מוגדר")
    if _index_state["error"]:
        raise HTTPException(500, f"שגיאה בבניית האינדקס: {_index_state['error']}")
    if not _index_state["ready"]:
        raise HTTPException(503, "האינדקס בתהליך בנייה — נסה שוב בעוד דקה")
    sources = sorted({c.source for c in index.chunks})
    return IndexStatus(
        documents=len({c.source for c in index.chunks}),
        chunks=len(index.chunks),
        sources=sources,
    )


@app.post("/api/reindex", response_model=IndexStatus)
def reindex() -> IndexStatus:
    if index is None:
        raise HTTPException(503, "OPENAI_API_KEY לא מוגדר")
    summary = index.build(KNOWLEDGE_DIR)
    _index_state["ready"] = True
    _index_state["error"] = None
    print(
        f"[main] reindex: {summary['documents']} מסמכים, "
        f"{summary['chunks']} chunks ({summary['new']} חדשים, {summary['cached']} מ-cache)."
    )
    return IndexStatus(
        documents=summary["documents"],
        chunks=summary["chunks"],
        sources=summary["sources"],
    )


@app.post("/api/session/reset")
def reset_session(session_id: str | None = None) -> dict:
    """איפוס היסטוריית שיחה — מתחיל סשן חדש."""
    if session_id:
        _sessions.pop(session_id, None)
        db.delete_session(session_id)
    new_id = str(uuid.uuid4())
    _sessions[new_id] = ChatSession()
    return {"session_id": new_id, "history_size": 0}


@app.post("/api/upload")
def upload_files(files: list[UploadFile] = File(...)) -> dict:
    """מקבל קבצי ידע, שומר אותם ומפעיל reindex."""
    if index is None:
        raise HTTPException(503, "OPENAI_API_KEY לא מוגדר")

    saved, skipped, errors = [], [], []
    KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)

    for f in files:
        suffix = Path(f.filename or "").suffix.lower()
        if suffix not in ALLOWED_EXTENSIONS:
            skipped.append(f.filename)
            continue
        dest = KNOWLEDGE_DIR / f.filename
        try:
            with dest.open("wb") as out:
                shutil.copyfileobj(f.file, out)
            saved.append(f.filename)
        except Exception as e:
            errors.append(f"{f.filename}: {e}")
        finally:
            f.file.close()

    if not saved:
        return {"saved": [], "skipped": skipped, "errors": errors,
                "documents": 0, "chunks": 0}

    # reindex אחרי שמירה
    summary = index.build(KNOWLEDGE_DIR)
    _index_state["ready"] = True
    print(f"[upload] נשמרו {len(saved)} קבצים, reindex → {summary['documents']} מסמכים")
    return {
        "saved": saved,
        "skipped": skipped,
        "errors": errors,
        "documents": summary["documents"],
        "chunks": summary["chunks"],
    }


@app.delete("/api/upload/{filename}")
def delete_file(filename: str) -> dict:
    """מוחק קובץ מהמאגר ומריץ reindex."""
    if index is None:
        raise HTTPException(503, "OPENAI_API_KEY לא מוגדר")
    target = KNOWLEDGE_DIR / filename
    if not target.exists():
        raise HTTPException(404, f"קובץ לא נמצא: {filename}")
    # מניעת path traversal
    try:
        target.resolve().relative_to(KNOWLEDGE_DIR.resolve())
    except ValueError:
        raise HTTPException(400, "שם קובץ לא חוקי")
    target.unlink()
    summary = index.build(KNOWLEDGE_DIR)
    _index_state["ready"] = True
    return {"deleted": filename, "documents": summary["documents"], "chunks": summary["chunks"]}


@app.get("/api/files")
def list_files() -> dict:
    """מחזיר רשימת קבצים במאגר."""
    KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)
    files = []
    for p in sorted(KNOWLEDGE_DIR.rglob("*")):
        if p.is_file() and p.suffix.lower() in ALLOWED_EXTENSIONS:
            files.append({
                "name": p.name,
                "size": p.stat().st_size,
                "modified": p.stat().st_mtime,
            })
    return {"files": files}


@app.post("/api/feedback")
def save_feedback(req: FeedbackRequest) -> dict:
    """שומר פידבק 👍/👎 על תשובה."""
    if req.rating not in (1, -1):
        raise HTTPException(400, "rating חייב להיות 1 או -1")
    db.save_feedback(
        session_id=req.session_id or "",
        question=req.question,
        answer_preview=req.answer_preview,
        rating=req.rating,
    )
    return {"ok": True}


@app.post("/api/chat/stream")
def chat_stream(req: ChatRequest):
    """תשובה ב-streaming. מחזיר Server-Sent Events.

    מבנה ההודעות:
      data: {"type": "session", "session_id": "..."}
      data: {"type": "sources", "matches": [...]}
      data: {"type": "delta", "text": "..."}
      data: {"type": "done"}
    """
    if index is None or client is None:
        raise HTTPException(503, "OPENAI_API_KEY לא מוגדר")
    if not _index_state["ready"]:
        raise HTTPException(503, "המאגר עדיין נטען — נסה שוב בעוד דקה")
    if not req.question.strip():
        raise HTTPException(400, "שאלה ריקה")

    session_id, session = _get_session(req.session_id)
    results = index.search(req.question, k=10)

    def event_stream():
        yield f"data: {json.dumps({'type': 'session', 'session_id': session_id})}\n\n"

        matches = [
            {
                "source": r.source,
                "section": r.section,
                "score": r.score,
                "vector_rank": r.vector_rank,
                "bm25_rank": r.bm25_rank,
                "preview": r.text[:250],
            }
            for r in results
        ]
        yield f"data: {json.dumps({'type': 'sources', 'matches': matches}, ensure_ascii=False)}\n\n"

        # --- צריכת תשובת ה-LLM עם keepalive ---
        # gpt-5.x "חושב" שניות רבות לפני הטוקן הראשון, ובמהלך השיחה זה
        # מתארך. בזמן השקט הזה אין נתונים ב-SSE — מה שגורם ל-proxy של
        # Railway/הרשת לסגור את החיבור ("network error" אצל המשתמש).
        # פתרון: מריצים את הסטרים של OpenAI ב-thread נפרד, ושולחים הערת
        # SSE (": ping") כל כמה שניות כל עוד אין delta — כך החיבור נשאר חי.
        full_text = ""
        delta_q: queue.Queue = queue.Queue()
        _DONE = object()
        producer_state: dict = {"error": None, "value": ""}

        def _produce():
            gen = stream_answer(client, req.question, results, session)
            try:
                while True:
                    try:
                        delta_q.put(next(gen))
                    except StopIteration as stop:
                        producer_state["value"] = stop.value or ""
                        break
            except Exception as exc:  # שגיאת OpenAI באמצע הסטרים
                producer_state["error"] = str(exc)
            finally:
                delta_q.put(_DONE)

        threading.Thread(target=_produce, daemon=True).start()

        KEEPALIVE_SEC = 10
        while True:
            try:
                item = delta_q.get(timeout=KEEPALIVE_SEC)
            except queue.Empty:
                # אין delta כבר 10 שניות — שולחים הערת SSE כדי לשמור על החיבור
                yield ": ping\n\n"
                continue
            if item is _DONE:
                break
            full_text += item
            yield f"data: {json.dumps({'type': 'delta', 'text': item}, ensure_ascii=False)}\n\n"

        # אם לא הגיע אף delta אבל יש ערך סופי (מקרה קצה)
        if not full_text and producer_state["value"]:
            full_text = producer_state["value"]
            yield f"data: {json.dumps({'type': 'delta', 'text': full_text}, ensure_ascii=False)}\n\n"

        # אם הייתה שגיאה בצד OpenAI — מדווחים למשתמש בצורה ידידותית
        if producer_state["error"] and not full_text:
            yield f"data: {json.dumps({'type': 'error', 'message': 'אירעה שגיאה זמנית מול מנוע הבינה. נסה/י שוב.'}, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
            return

        # שמור את ההיסטוריה
        session.add("user", req.question)
        session.add("assistant", full_text)
        db.save_session(session_id, session.to_dict_list())

        # שאלות המשך
        if full_text:
            followups = generate_followups(client, req.question, full_text)
            if followups:
                yield f"data: {json.dumps({'type': 'followups', 'questions': followups}, ensure_ascii=False)}\n\n"

        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # מנטרל buffering של nginx
        },
    )


# סטטיים
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
