"""FastAPI app — בוט מענה אישי עם hybrid search, conversation memory, ו-streaming."""
from __future__ import annotations

import json
import os
import threading
import traceback
import uuid
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from openai import OpenAI
from pydantic import BaseModel

from llm import ChatSession, MAX_HISTORY_TURNS, stream_answer
from rag import HybridIndex

# --- הגדרות ---
BACKEND_DIR = Path(__file__).parent
PROJECT_DIR = BACKEND_DIR.parent
KNOWLEDGE_DIR = BACKEND_DIR / "data" / "knowledge"
CACHE_PATH = BACKEND_DIR / "data" / ".embed_cache.pkl"
FRONTEND_DIR = PROJECT_DIR / "frontend"

load_dotenv(PROJECT_DIR / ".env")

API_KEY = os.environ.get("OPENAI_API_KEY")
if not API_KEY:
    print("[main] אזהרה: לא הוגדר OPENAI_API_KEY. הבוט לא יענה עד שיוגדר.")

client = OpenAI(api_key=API_KEY) if API_KEY else None
index = HybridIndex(client=client, cache_path=CACHE_PATH) if client else None

# Sessions בזיכרון. בפרודקשן צריך Redis או DB.
_sessions: dict[str, ChatSession] = {}

# סטטוס בניית האינדקס — כדי שהשרת יוכל לדווח אם האינדקס מוכן.
_index_state = {"ready": False, "building": False, "error": None}


def _get_session(session_id: str | None) -> tuple[str, ChatSession]:
    if session_id and session_id in _sessions:
        return session_id, _sessions[session_id]
    new_id = str(uuid.uuid4())
    _sessions[new_id] = ChatSession()
    return new_id, _sessions[new_id]


# --- מודלים ---


class ChatRequest(BaseModel):
    question: str
    session_id: str | None = None


class IndexStatus(BaseModel):
    documents: int
    chunks: int
    sources: list[str]


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
    if session_id and session_id in _sessions:
        del _sessions[session_id]
    new_id = str(uuid.uuid4())
    _sessions[new_id] = ChatSession()
    return {"session_id": new_id, "history_size": 0}


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

        full_text = ""
        gen = stream_answer(client, req.question, results, session)
        try:
            while True:
                delta = next(gen)
                full_text += delta
                yield f"data: {json.dumps({'type': 'delta', 'text': delta}, ensure_ascii=False)}\n\n"
        except StopIteration as stop:
            if not full_text and stop.value:
                full_text = stop.value
                yield f"data: {json.dumps({'type': 'delta', 'text': full_text}, ensure_ascii=False)}\n\n"

        # שמור את ההיסטוריה
        session.add("user", req.question)
        session.add("assistant", full_text)

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
