"""Retrieval-augmented generation layer over the turbofan corpus.

Two interchangeable embedding backends:

* ``openai``  — text-embedding-3-small (1536-d). Production path. Requires
  OPENAI_API_KEY in the environment.
* ``tfidf``   — sklearn TfidfVectorizer with cosine similarity. Zero-dependency
  baseline used when no API key is set. Useful for offline evaluation and
  unit tests; slightly weaker than dense retrieval on paraphrased queries.

Both backends expose the same Retriever interface and serialise to disk so the
HF Space can load the index without recomputing embeddings.
"""
from __future__ import annotations

import json
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

OPENAI_EMBED_MODEL = "text-embedding-3-small"
OPENAI_EMBED_DIM = 1536


# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------

@dataclass
class Doc:
    id: str
    title: str
    source: str
    text: str

    @property
    def embed_text(self) -> str:
        # Title is duplicated into the embedding text to bias toward topical
        # questions (small but consistent improvement on the eval set).
        return f"{self.title}\n\n{self.text}"


def load_corpus(path: str | Path) -> list[Doc]:
    docs: list[Doc] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            docs.append(Doc(id=obj["id"], title=obj["title"],
                            source=obj["source"], text=obj["text"]))
    if len({d.id for d in docs}) != len(docs):
        raise ValueError("corpus has duplicate ids")
    return docs


# ---------------------------------------------------------------------------
# Embedding backends
# ---------------------------------------------------------------------------

class _OpenAIBackend:
    name = "openai"

    def __init__(self) -> None:
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError(
                "OPENAI_API_KEY not set; cannot use OpenAI embedding backend."
            )
        from openai import OpenAI
        self._client = OpenAI()
        self.dim = OPENAI_EMBED_DIM

    def embed(self, texts: list[str]) -> np.ndarray:
        # OpenAI accepts up to 2048 inputs per call; our corpus is small.
        out: list[list[float]] = []
        for i in range(0, len(texts), 256):
            chunk = texts[i : i + 256]
            resp = self._client.embeddings.create(model=OPENAI_EMBED_MODEL, input=chunk)
            out.extend([d.embedding for d in resp.data])
        return _l2_normalize(np.asarray(out, dtype=np.float32))


class _TfidfBackend:
    name = "tfidf"

    def __init__(self) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer
        self._vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=1,
            sublinear_tf=True,
            stop_words="english",
        )
        self.dim: int | None = None  # set after fit
        self._fitted = False

    def fit(self, texts: list[str]) -> None:
        self._vectorizer.fit(texts)
        self.dim = len(self._vectorizer.vocabulary_)
        self._fitted = True

    def embed(self, texts: list[str]) -> np.ndarray:
        if not self._fitted:
            self.fit(texts)
        mat = self._vectorizer.transform(texts).astype(np.float32).toarray()
        return _l2_normalize(mat)


def make_backend(name: str | None = None) -> _OpenAIBackend | _TfidfBackend:
    """Pick a backend. If ``name`` is None, prefer OpenAI when key is set."""
    if name is None:
        name = "openai" if os.getenv("OPENAI_API_KEY") else "tfidf"
    if name == "openai":
        return _OpenAIBackend()
    if name == "tfidf":
        return _TfidfBackend()
    raise ValueError(f"unknown backend: {name!r}")


# ---------------------------------------------------------------------------
# Index build & retrieve
# ---------------------------------------------------------------------------

@dataclass
class Hit:
    doc: Doc
    score: float
    rank: int

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.doc.id, "title": self.doc.title,
                "source": self.doc.source, "score": float(self.score),
                "rank": self.rank, "text": self.doc.text}


class Retriever:
    """In-memory FAISS retriever (IndexFlatIP over L2-normalised vectors)."""

    def __init__(self, docs: list[Doc], embeddings: np.ndarray,
                 backend_name: str, backend: Any | None = None) -> None:
        import faiss
        if embeddings.shape[0] != len(docs):
            raise ValueError("docs and embeddings count mismatch")
        self.docs = docs
        self.embeddings = embeddings.astype(np.float32, copy=False)
        self.backend_name = backend_name
        self._backend = backend
        self._index = faiss.IndexFlatIP(embeddings.shape[1])
        self._index.add(self.embeddings)

    @classmethod
    def build(cls, docs: list[Doc], backend_name: str | None = None) -> "Retriever":
        backend = make_backend(backend_name)
        if isinstance(backend, _TfidfBackend):
            backend.fit([d.embed_text for d in docs])
        embeddings = backend.embed([d.embed_text for d in docs])
        return cls(docs=docs, embeddings=embeddings,
                   backend_name=backend.name, backend=backend)

    def retrieve(self, query: str, k: int = 5) -> list[Hit]:
        if self._backend is None:
            raise RuntimeError("retriever loaded without backend; call attach_backend()")
        q_emb = self._backend.embed([query])
        scores, idx = self._index.search(q_emb, k)
        hits: list[Hit] = []
        for rank, (i, s) in enumerate(zip(idx[0].tolist(), scores[0].tolist()), start=1):
            if i < 0:
                continue
            hits.append(Hit(doc=self.docs[i], score=float(s), rank=rank))
        return hits

    # ---- persistence -------------------------------------------------------

    def save(self, dir_path: str | Path) -> None:
        dir_path = Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)
        np.save(dir_path / "embeddings.npy", self.embeddings)
        with open(dir_path / "docs.jsonl", "w", encoding="utf-8") as f:
            for d in self.docs:
                f.write(json.dumps({"id": d.id, "title": d.title,
                                     "source": d.source, "text": d.text}) + "\n")
        meta = {"backend": self.backend_name, "n_docs": len(self.docs),
                "dim": int(self.embeddings.shape[1])}
        (dir_path / "meta.json").write_text(json.dumps(meta, indent=2))
        if isinstance(self._backend, _TfidfBackend):
            with open(dir_path / "tfidf.pkl", "wb") as f:
                pickle.dump(self._backend._vectorizer, f)

    @classmethod
    def load(cls, dir_path: str | Path,
             override_backend: str | None = None) -> "Retriever":
        dir_path = Path(dir_path)
        meta = json.loads((dir_path / "meta.json").read_text())
        embeddings = np.load(dir_path / "embeddings.npy")
        docs: list[Doc] = []
        with open(dir_path / "docs.jsonl", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                docs.append(Doc(**obj))
        backend_name = override_backend or meta["backend"]
        if backend_name == "tfidf":
            backend = _TfidfBackend()
            with open(dir_path / "tfidf.pkl", "rb") as f:
                backend._vectorizer = pickle.load(f)
            backend._fitted = True
            backend.dim = len(backend._vectorizer.vocabulary_)
        elif backend_name == "openai":
            backend = _OpenAIBackend()
        else:
            raise ValueError(f"unknown backend: {backend_name!r}")
        return cls(docs=docs, embeddings=embeddings,
                   backend_name=backend_name, backend=backend)


def _l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, eps)
