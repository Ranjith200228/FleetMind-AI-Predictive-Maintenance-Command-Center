"""Build the RAG index from data/rag/corpus.jsonl.

Usage:
  python -m scripts.build_index --backend tfidf
  OPENAI_API_KEY=... python -m scripts.build_index --backend openai
  python -m scripts.build_index           # auto: openai if key set, else tfidf
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.rag import Retriever, load_corpus  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", default="data/rag/corpus.jsonl")
    p.add_argument("--out", default="data/rag_index")
    p.add_argument("--backend", default=None,
                   choices=["openai", "tfidf"],
                   help="auto = openai if OPENAI_API_KEY else tfidf")
    args = p.parse_args()

    docs = load_corpus(args.corpus)
    print(f"[index] loaded {len(docs)} docs from {args.corpus}")

    t0 = time.time()
    retriever = Retriever.build(docs, backend_name=args.backend)
    print(f"[index] embedded with backend={retriever.backend_name}  "
          f"dim={retriever.embeddings.shape[1]}  ({time.time()-t0:.1f}s)")

    out = Path(args.out)
    retriever.save(out)
    print(f"[index] saved to {out}/")


if __name__ == "__main__":
    main()
