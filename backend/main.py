"""FastAPI app — backend לבוט מענה אישי."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from openai import OpenAI
from pydantic import BaseModel

from llm import answer_question
from rag import VectorIndex

# --- הגדרות ---
BACKEND_DIR = Path(__file__).parent
PROJECT_DIR = BACKEND_DIR.parent
KNOWLEDGE_DIR = BACKEND_DIR / "data" / "knowledge"
CACHE_PATH = BACKEND_DIR / "data" / ".embed_cache.pkl"
FRONTEND_DIR = PROJECT_DIR / "frontend"

load_dotenv(PROJECT_DIR / ".env")

API_KEY = os.environ.get("OPENAI_API_KEY")
if not API_KEY:
    print("[main] אזהרה: לא הוגדר OPENAI_API_KEY. הבוט לא יוכל לענות עד שיוגדר.")

client = OpenAI(api_key=API_KEY) if API_KEY else None
index = VectorIndex(client=client, cache_path=CACHE_PATH) if client else None

# --- מודלים של בקשות/תשובות ---


class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[str]
    matches: list[dict]


class IndexStatus(BaseModel):
    documents: int
    chunks: int
    sources: list[str]


# --- אפליקציה ---
app = FastAPI(title="בוט מענה אישי", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup() -> None:
    if index is None:
        return
    KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)
    summary = index.build(KNOWLEDGE_DIR)
    print(
        f"[main] נטענו {summary['documents']} מסמכים, "
        f"{summary['chunks']} chunks ({summary['new']} חדשים, {summary['cached']} מ-cache)."
    )


@app.get("/")
def root() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/api/status", response_model=IndexStatus)
def status() -> IndexStatus:
    if index is None:
        raise HTTPException(503, "OPENAI_API_KEY לא מוגדר")
    sources = sorted({c.source for c in index.chunks})
    return IndexStatus(
        documents=len({c.source for c in index.chunks}),
        chunks=len(index.chunks),
        sources=sources,
    )


@app.post("/api/reindex", response_model=IndexStatus)
def reindex() -> IndexStatus:
    """לקרוא מחדש את כל המסמכים מהתיקייה (אחרי הוספת/שינוי מסמך)."""
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


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    if index is None or client is None:
        raise HTTPException(503, "OPENAI_API_KEY לא מוגדר")
    if not req.question.strip():
        raise HTTPException(400, "שאלה ריקה")
    results = index.search(req.question, k=5)
    ans = answer_question(client, req.question, results)
    return ChatResponse(
        answer=ans.answer,
        sources=ans.sources,
        matches=[
            {"source": r.source, "score": round(r.score, 3), "preview": r.text[:200]}
            for r in results
        ],
    )


# נתיב סטטי לקבצים בפרונט
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
