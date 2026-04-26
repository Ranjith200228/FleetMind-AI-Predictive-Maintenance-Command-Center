"""FleetMind Copilot agent: one tool, grounded by RAG citations.

Loop:
  1. User asks a question.
  2. Retrieve top-k corpus chunks for the question (always).
  3. Send {system, retrieved_context, user} to OpenAI with one tool exposed:
     query_engine_history.
  4. If the model emits a tool call, execute it, append the tool result, and
     get the model's final answer.
  5. Return {answer, citations, tool_calls} for the UI.

Without OPENAI_API_KEY the agent falls back to a deterministic mock that runs
retrieval and surfaces the top context — useful for offline smoke tests.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any

from src.rag import Hit, Retriever
from src.tools import TOOL_SCHEMA, EngineDataBackend, call_tool, query_engine_history

DEFAULT_MODEL = "gpt-4o-mini"
TOP_K = 5

SYSTEM_PROMPT = """You are FleetMind Copilot, an assistant for predictive \
maintenance of C-MAPSS turbofan engines.

TOOL: You have exactly one tool, `query_engine_history(engine_id)`. Call it \
ONLY when the user references a specific engine id (1-100) or asks about \
the current status, RUL, or recommended action for a particular engine. For \
general questions about the dataset, fault modes, sensors, baselines, or \
modelling, do NOT call the tool — answer from the retrieved context only.

CITATIONS — STRICT: Every factual sentence in your answer MUST end with one \
or more [doc_id] markers identifying the retrieved chunk that supports it. \
The valid doc_ids are exactly the bracketed ids you see in RETRIEVED \
CONTEXT (e.g. [cmapss-overview], [rul-piecewise]). Do not invent doc_ids. \
If you cannot find support for a claim in the retrieved context, drop the \
claim or say so explicitly. Tool-output sentences (specific RUL value, \
recommended action) do not need a citation, but the surrounding domain \
explanation does. An answer with zero [doc_id] markers is wrong.

LENGTH: 4-8 sentences for general questions; longer only when reporting \
tool output.

WHEN YOU RECEIVE TOOL OUTPUT for a specific engine, ground your maintenance \
recommendation in: (a) the LSTM RUL prediction, (b) the recommended_action \
band returned by the tool, and (c) the relevant sensor trend (rising T30 / \
T50 / P30 / Ps30 / W31 / W32 indicates HPC degradation on FD001). Add at \
least one [doc_id] citation from the retrieved context to back the action."""


@dataclass
class AgentResult:
    answer: str
    citations: list[dict[str, Any]]
    tool_calls: list[dict[str, Any]]
    retrieved: list[dict[str, Any]]
    mode: str  # "openai" or "mock"

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "citations": self.citations,
            "tool_calls": self.tool_calls,
            "retrieved": self.retrieved,
            "mode": self.mode,
        }


class FleetMindAgent:
    def __init__(
        self,
        retriever: Retriever,
        engine_backend: EngineDataBackend,
        model: str = DEFAULT_MODEL,
        top_k: int = TOP_K,
    ) -> None:
        self.retriever = retriever
        self.engine_backend = engine_backend
        self.model = model
        self.top_k = top_k
        self._client = None
        if os.getenv("OPENAI_API_KEY"):
            from openai import OpenAI
            self._client = OpenAI()

    # ------------------------------------------------------------------ chat

    def chat(self, user_msg: str) -> AgentResult:
        hits = self.retriever.retrieve(user_msg, k=self.top_k)
        if self._client is None:
            return self._mock_chat(user_msg, hits)
        return self._openai_chat(user_msg, hits)

    # ----------------------------------------------------------- OpenAI path

    def _openai_chat(self, user_msg: str, hits: list[Hit]) -> AgentResult:
        context = _format_context(hits)
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "system", "content": f"RETRIEVED CONTEXT:\n\n{context}"},
            {"role": "user", "content": user_msg},
        ]
        tool_calls_log: list[dict[str, Any]] = []

        first = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=[TOOL_SCHEMA],
            tool_choice="auto",
            temperature=0.2,
        )
        msg = first.choices[0].message
        # Append the assistant message to keep tool_call_id linkage valid.
        messages.append({
            "role": "assistant",
            "content": msg.content,
            "tool_calls": [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name,
                              "arguments": tc.function.arguments}}
                for tc in (msg.tool_calls or [])
            ],
        })

        if msg.tool_calls:
            for tc in msg.tool_calls:
                args = tc.function.arguments
                result_json = call_tool(tc.function.name, args, self.engine_backend)
                tool_calls_log.append({
                    "name": tc.function.name,
                    "arguments": json.loads(args) if isinstance(args, str) else args,
                    "result": json.loads(result_json),
                })
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result_json,
                })
            second = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.2,
            )
            answer = second.choices[0].message.content or ""
        else:
            answer = msg.content or ""

        cited_ids = _extract_citations(answer)
        citations = [h.to_dict() for h in hits if h.doc.id in cited_ids]
        return AgentResult(
            answer=answer,
            citations=citations,
            tool_calls=tool_calls_log,
            retrieved=[h.to_dict() for h in hits],
            mode="openai",
        )

    # ------------------------------------------------------------- mock path

    def _mock_chat(self, user_msg: str, hits: list[Hit]) -> AgentResult:
        """Deterministic offline fallback. Surfaces retrieval + tool result.

        If the user mentions an engine id, runs the tool and includes the
        prediction + action band. Otherwise returns the top-1 retrieval
        snippet with its citation.
        """
        engine_id = _extract_engine_id(user_msg)
        tool_calls_log: list[dict[str, Any]] = []
        if engine_id is not None:
            result = query_engine_history(engine_id, self.engine_backend)
            tool_calls_log.append({
                "name": "query_engine_history",
                "arguments": {"engine_id": engine_id},
                "result": result,
            })
            if result.get("ok"):
                p = result["lstm_prediction"]
                top = hits[0]
                answer = (
                    f"Engine {engine_id}: predicted RUL {p['rul_cycles']} cycles. "
                    f"Recommended action: {p['recommended_action']}. "
                    f"{p['action_rationale']} [{top.doc.id}] "
                    f"(MOCK MODE — no OPENAI_API_KEY set; this answer is deterministic.)"
                )
                citations = [top.to_dict()]
            else:
                answer = result.get("error", "tool error")
                citations = []
        else:
            top = hits[0]
            answer = (
                f"{top.doc.text} [{top.doc.id}]\n\n"
                f"(MOCK MODE — no OPENAI_API_KEY set; returning top retrieval verbatim.)"
            )
            citations = [top.to_dict()]
        return AgentResult(
            answer=answer,
            citations=citations,
            tool_calls=tool_calls_log,
            retrieved=[h.to_dict() for h in hits],
            mode="mock",
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_context(hits: list[Hit]) -> str:
    blocks = []
    for h in hits:
        blocks.append(
            f"[{h.doc.id}] {h.doc.title} — source: {h.doc.source}\n{h.doc.text}"
        )
    return "\n\n".join(blocks)


_CITATION_RE = re.compile(r"\[([a-z0-9][a-z0-9\-]*)\]")
_ENGINE_ID_RE = re.compile(r"\bengine[\s_]*#?\s*(\d{1,3})\b", re.IGNORECASE)


def _extract_citations(text: str) -> set[str]:
    return set(_CITATION_RE.findall(text))


def _extract_engine_id(text: str) -> int | None:
    m = _ENGINE_ID_RE.search(text)
    if not m:
        return None
    eid = int(m.group(1))
    if 1 <= eid <= 100:
        return eid
    return None
