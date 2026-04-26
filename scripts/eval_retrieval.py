"""Evaluate the RAG retriever on the held-out 50-pair Q/A set.

Reports hit@1, hit@3, hit@5, hit@10, MRR@10. Writes
reports/retrieval_metrics.json with per-query results for transparency.

Usage:
  python -m scripts.eval_retrieval [--index data/rag_index] [--qa data/rag/eval_qa.jsonl]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.rag import Retriever  # noqa: E402


def evaluate(retriever: Retriever, qa: list[dict], ks: tuple[int, ...] = (1, 3, 5, 10)) -> dict:
    k_max = max(ks)
    hits = {k: 0 for k in ks}
    rr_total = 0.0
    per_query = []
    for ex in qa:
        gold = set(ex["answers"])
        results = retriever.retrieve(ex["q"], k=k_max)
        ranked_ids = [h.doc.id for h in results]
        # reciprocal rank of first gold hit, 0 if none in top-k_max
        rr = 0.0
        first_rank = None
        for rank, rid in enumerate(ranked_ids, start=1):
            if rid in gold:
                rr = 1.0 / rank
                first_rank = rank
                break
        rr_total += rr
        for k in ks:
            if any(rid in gold for rid in ranked_ids[:k]):
                hits[k] += 1
        per_query.append({
            "q": ex["q"],
            "gold": sorted(gold),
            "top1": ranked_ids[0] if ranked_ids else None,
            "first_gold_rank": first_rank,
            "rr": rr,
            "top_scores": [round(h.score, 4) for h in results[:5]],
        })
    n = len(qa)
    return {
        "n": n,
        "hit@1": hits[1] / n,
        "hit@3": hits[3] / n,
        "hit@5": hits[5] / n,
        "hit@10": hits[10] / n,
        "mrr@10": rr_total / n,
        "per_query": per_query,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--index", default="data/rag_index")
    p.add_argument("--qa", default="data/rag/eval_qa.jsonl")
    p.add_argument("--out", default="reports/retrieval_metrics.json")
    p.add_argument("--backend", default=None,
                   help="override backend at load time (advanced)")
    args = p.parse_args()

    retriever = Retriever.load(args.index, override_backend=args.backend)
    qa = [json.loads(l) for l in open(args.qa, encoding="utf-8") if l.strip()]
    print(f"[eval] backend={retriever.backend_name}  docs={len(retriever.docs)}  qa={len(qa)}")

    t0 = time.time()
    metrics = evaluate(retriever, qa)
    metrics["backend"] = retriever.backend_name
    metrics["n_docs"] = len(retriever.docs)
    metrics["seconds"] = round(time.time() - t0, 1)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2))
    print(f"[eval] hit@1={metrics['hit@1']:.3f}  hit@3={metrics['hit@3']:.3f}  "
          f"hit@5={metrics['hit@5']:.3f}  hit@10={metrics['hit@10']:.3f}  "
          f"mrr@10={metrics['mrr@10']:.3f}  ({metrics['seconds']}s)")
    print(f"[eval] wrote {out}")


if __name__ == "__main__":
    main()
