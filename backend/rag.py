"""RAG pipeline: embeddings ב-OpenAI + חיפוש קוסינוס + cache בדיסק."""
from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from openai import OpenAI

from loaders import build_chunks, load_documents

EMBED_MODEL = "text-embedding-3-small"  # זול, איכותי בעברית
EMBED_DIM = 1536
BATCH_SIZE = 100


@dataclass
class Chunk:
    source: str
    text: str
    embedding: np.ndarray  # shape (EMBED_DIM,)


@dataclass
class SearchResult:
    source: str
    text: str
    score: float


def _hash_chunk(source: str, text: str) -> str:
    h = hashlib.sha256()
    h.update(source.encode("utf-8"))
    h.update(b"\x00")
    h.update(text.encode("utf-8"))
    h.update(b"\x00")
    h.update(EMBED_MODEL.encode("utf-8"))
    return h.hexdigest()


class VectorIndex:
    """אינדקס וקטורי פשוט בזיכרון, עם cache בדיסק לפי hash של כל chunk."""

    def __init__(self, client: OpenAI, cache_path: Path):
        self.client = client
        self.cache_path = cache_path
        self.chunks: list[Chunk] = []
        self._matrix: np.ndarray | None = None
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
        """קורא את כל המסמכים מהתיקייה ובונה אינדקס. מחזיר סיכום."""
        docs = load_documents(knowledge_dir)
        pairs = build_chunks(docs)

        if not pairs:
            self.chunks = []
            self._matrix = None
            return {
                "documents": 0,
                "chunks": 0,
                "cached": 0,
                "new": 0,
                "sources": [],
            }

        # מצא chunks שלא ב-cache
        to_embed_idx: list[int] = []
        to_embed_texts: list[str] = []
        hashes: list[str] = []
        for i, (source, text) in enumerate(pairs):
            key = _hash_chunk(source, text)
            hashes.append(key)
            if key not in self._embed_cache:
                to_embed_idx.append(i)
                to_embed_texts.append(text)

        # embed בבאצ'ים
        for start in range(0, len(to_embed_texts), BATCH_SIZE):
            batch = to_embed_texts[start : start + BATCH_SIZE]
            embeddings = self._embed_batch(batch)
            for j, emb in enumerate(embeddings):
                idx_in_pairs = to_embed_idx[start + j]
                key = hashes[idx_in_pairs]
                self._embed_cache[key] = emb

        self._save_cache()

        # בנה את האינדקס
        self.chunks = []
        vectors = []
        for (source, text), key in zip(pairs, hashes):
            emb = self._embed_cache[key]
            self.chunks.append(Chunk(source=source, text=text, embedding=emb))
            vectors.append(emb)

        # נרמול לחישוב cosine מהיר עם dot product
        matrix = np.vstack(vectors)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self._matrix = matrix / norms

        sources_set = sorted({c.source for c in self.chunks})
        return {
            "documents": len(docs),
            "chunks": len(self.chunks),
            "cached": len(pairs) - len(to_embed_texts),
            "new": len(to_embed_texts),
            "sources": sources_set,
        }

    def search(self, query: str, k: int = 5) -> list[SearchResult]:
        if self._matrix is None or not self.chunks:
            return []
        q_emb = self._embed_batch([query])[0]
        q_norm = np.linalg.norm(q_emb) or 1.0
        q_vec = q_emb / q_norm
        scores = self._matrix @ q_vec
        top_idx = np.argsort(-scores)[:k]
        results = []
        for i in top_idx:
            results.append(
                SearchResult(
                    source=self.chunks[i].source,
                    text=self.chunks[i].text,
                    score=float(scores[i]),
                )
            )
        return results
