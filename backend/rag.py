"""RAG pipeline: Hybrid search (Vector + BM25) עם Reciprocal Rank Fusion.

- Embeddings: text-embedding-3-large (3072 dim) — איכות גבוהה יותר בעברית
- Vector search: cosine similarity
- BM25: חיפוש מילולי — חשוב לראשי תיבות, שמות, מספרים
- RRF: שילוב התוצאות בשיטה standard (k=60)
- Cache: hash תוכן + שם מודל בדיסק
"""
from __future__ import annotations

import hashlib
import pickle
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from openai import OpenAI
from rank_bm25 import BM25Okapi

from loaders import Chunk, build_chunks, load_documents

EMBED_MODEL = "text-embedding-3-large"  # 3072-dim, איכות גבוהה בעברית
BATCH_SIZE = 100
RRF_K = 60  # קבוע סטנדרטי ל-Reciprocal Rank Fusion
RETRIEVE_PER_METHOD = 20  # כמה תוצאות לקחת מכל שיטה לפני שילוב


@dataclass
class IndexedChunk:
    chunk: Chunk
    embedding: np.ndarray


@dataclass
class SearchResult:
    source: str
    text: str
    section: str
    score: float
    vector_rank: int | None = None
    bm25_rank: int | None = None


def _hash_chunk(source: str, text: str) -> str:
    h = hashlib.sha256()
    h.update(source.encode("utf-8"))
    h.update(b"\x00")
    h.update(text.encode("utf-8"))
    h.update(b"\x00")
    h.update(EMBED_MODEL.encode("utf-8"))
    return h.hexdigest()


_HEBREW_TOKEN_RE = re.compile(r"[א-תA-Za-z0-9א-ת]+")


def _tokenize_hebrew(text: str) -> list[str]:
    """Tokenizer פשוט שעובד בעברית, אנגלית ומספרים. שומר על ראשי תיבות עם גרשיים."""
    # שמור על ראשי תיבות עם גרש/גרשיים (לדוגמה: קב"ס, מל"א, יו"ר)
    cleaned = re.sub(r'["\'״׳]', '', text)
    tokens = _HEBREW_TOKEN_RE.findall(cleaned.lower())
    # סנן tokens קצרים מאוד (1 תו)
    return [t for t in tokens if len(t) >= 2]


class HybridIndex:
    """אינדקס היברידי בזיכרון: Vector (OpenAI) + BM25, משולבים ב-RRF."""

    def __init__(self, client: OpenAI, cache_path: Path):
        self.client = client
        self.cache_path = cache_path
        self.chunks: list[Chunk] = []
        self._vector_matrix: np.ndarray | None = None  # נורמלי, שורה לכל chunk
        self._bm25: BM25Okapi | None = None
        self._embed_cache: dict[str, np.ndarray] = self._load_cache()

    def _load_cache(self) -> dict[str, np.ndarray]:
        if self.cache_path.exists():
            try:
                with self.cache_path.open("rb") as f:
                    return pickle.load(f)
            except Exception as exc:
                print(f"[rag] cache פגום, מתעלם: {exc}")
        return {}

    def _save_cache(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with self.cache_path.open("wb") as f:
            pickle.dump(self._embed_cache, f)

    def _embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        resp = self.client.embeddings.create(model=EMBED_MODEL, input=texts)
        return [np.array(d.embedding, dtype=np.float32) for d in resp.data]

    def build(self, knowledge_dir: Path) -> dict:
        """בונה את האינדקס מתיקיית הידע. מחזיר סיכום."""
        docs = load_documents(knowledge_dir)
        chunks = build_chunks(docs)

        if not chunks:
            self.chunks = []
            self._vector_matrix = None
            self._bm25 = None
            return {"documents": 0, "chunks": 0, "cached": 0, "new": 0, "sources": []}

        # --- חישוב embeddings (עם cache) ---
        hashes = [_hash_chunk(c.source, c.text) for c in chunks]
        to_embed_idx: list[int] = []
        to_embed_texts: list[str] = []
        for i, (chunk, key) in enumerate(zip(chunks, hashes)):
            if key not in self._embed_cache:
                to_embed_idx.append(i)
                to_embed_texts.append(chunk.text)

        for start in range(0, len(to_embed_texts), BATCH_SIZE):
            batch = to_embed_texts[start : start + BATCH_SIZE]
            embeddings = self._embed_batch(batch)
            for j, emb in enumerate(embeddings):
                key = hashes[to_embed_idx[start + j]]
                self._embed_cache[key] = emb
        self._save_cache()

        # --- בניית מטריצת וקטורים מנורמלת ---
        self.chunks = chunks
        vectors = np.vstack([self._embed_cache[h] for h in hashes])
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self._vector_matrix = vectors / norms

        # --- בניית BM25 ---
        tokenized = [_tokenize_hebrew(c.text) for c in chunks]
        self._bm25 = BM25Okapi(tokenized)

        sources_set = sorted({c.source for c in chunks})
        return {
            "documents": len({c.source for c in chunks}),
            "chunks": len(chunks),
            "cached": len(chunks) - len(to_embed_texts),
            "new": len(to_embed_texts),
            "sources": sources_set,
        }

    def _vector_search(self, query: str, k: int) -> list[tuple[int, float]]:
        """מחזיר רשימה של (chunk_idx, similarity) ממוין מהגבוה לנמוך."""
        if self._vector_matrix is None:
            return []
        q_emb = self._embed_batch([query])[0]
        q_norm = np.linalg.norm(q_emb) or 1.0
        q_vec = q_emb / q_norm
        scores = self._vector_matrix @ q_vec
        top_idx = np.argsort(-scores)[:k]
        return [(int(i), float(scores[i])) for i in top_idx]

    def _bm25_search(self, query: str, k: int) -> list[tuple[int, float]]:
        """מחזיר רשימה של (chunk_idx, bm25_score) ממוין מהגבוה לנמוך."""
        if self._bm25 is None:
            return []
        tokens = _tokenize_hebrew(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        top_idx = np.argsort(-scores)[:k]
        return [(int(i), float(scores[i])) for i in top_idx if scores[i] > 0]

    def search(self, query: str, k: int = 6) -> list[SearchResult]:
        """חיפוש היברידי: BM25 + Vector → RRF → top k."""
        if not self.chunks:
            return []

        vector_results = self._vector_search(query, RETRIEVE_PER_METHOD)
        bm25_results = self._bm25_search(query, RETRIEVE_PER_METHOD)

        # --- Reciprocal Rank Fusion ---
        rrf_scores: dict[int, float] = {}
        vec_ranks: dict[int, int] = {}
        bm25_ranks: dict[int, int] = {}

        for rank, (idx, _) in enumerate(vector_results, 1):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1.0 / (RRF_K + rank)
            vec_ranks[idx] = rank

        for rank, (idx, _) in enumerate(bm25_results, 1):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1.0 / (RRF_K + rank)
            bm25_ranks[idx] = rank

        # --- מיון לפי RRF score ובניית תוצאות ---
        sorted_idx = sorted(rrf_scores.keys(), key=lambda i: -rrf_scores[i])[:k]
        results: list[SearchResult] = []
        for idx in sorted_idx:
            chunk = self.chunks[idx]
            results.append(
                SearchResult(
                    source=chunk.source,
                    text=chunk.text,
                    section=chunk.section,
                    score=round(rrf_scores[idx], 4),
                    vector_rank=vec_ranks.get(idx),
                    bm25_rank=bm25_ranks.get(idx),
                )
            )
        return results
