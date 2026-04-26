"""Smoke-test the FleetMind agent on a few representative queries.

Runs in mock mode if OPENAI_API_KEY is not set, in real mode otherwise.
Writes reports/agent_demo.json with the full transcript so the run is
reproducible after the fact.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agent import FleetMindAgent  # noqa: E402
from src.rag import Retriever  # noqa: E402
from src.tools import EngineDataBackend  # noqa: E402

DEMO_QUERIES = [
    "What is the C-MAPSS dataset?",
    "Why is RUL clipped at 125 cycles in training?",
    "Which sensors react most strongly when the HPC degrades?",
    "What is the current RUL prediction for engine 17 and what should we do?",
    "Should we replace engine 100 immediately?",
    "Explain the C-MAPSS asymmetric scoring function and why late predictions are punished harder.",
]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--index", default="data/rag_index")
    p.add_argument("--out", default="reports/agent_demo.json")
    p.add_argument("--model", default="gpt-4o-mini")
    args = p.parse_args()

    print(f"[agent] OPENAI_API_KEY set: {bool(os.getenv('OPENAI_API_KEY'))}")
    retriever = Retriever.load(args.index)
    print(f"[agent] retriever backend={retriever.backend_name} docs={len(retriever.docs)}")
    engine_backend = EngineDataBackend.load()
    print(f"[agent] engine_backend test engines={len(engine_backend.engine_ids())}")

    agent = FleetMindAgent(retriever=retriever, engine_backend=engine_backend, model=args.model)
    transcript = []
    for q in DEMO_QUERIES:
        t0 = time.time()
        result = agent.chat(q)
        dt = time.time() - t0
        print(f"\n--- Q: {q}")
        print(f"    mode={result.mode}  cites={[c['id'] for c in result.citations]}  "
              f"tool_calls={[tc['name'] for tc in result.tool_calls]}  ({dt:.1f}s)")
        print(f"    A: {result.answer[:300]}{'...' if len(result.answer) > 300 else ''}")
        transcript.append({
            "q": q,
            "mode": result.mode,
            "answer": result.answer,
            "citations": [c["id"] for c in result.citations],
            "tool_calls": result.tool_calls,
            "seconds": round(dt, 2),
        })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"mode": agent._client and "openai" or "mock",
                                "model": args.model,
                                "transcript": transcript}, indent=2))
    print(f"\n[agent] wrote {out}")


if __name__ == "__main__":
    main()
