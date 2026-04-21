"""Refine REVIVE's swarm draft with Sonnet 4.6.

Async generator that yields text deltas suitable for feeding the same streaming
path the dashboard already consumes.
"""
from __future__ import annotations

from typing import AsyncIterator

from .bedrock import BedrockStreamer

_SYSTEM = (
    "You are REVIVE, a distributed-inference assistant. A local swarm of small "
    "models produced a draft answer to the user's question. Rewrite the draft "
    "into a clear, correct, concise reply. If the draft is incoherent or wrong, "
    "ignore it and answer the question directly from scratch. Do not mention the "
    "draft, the swarm, or that you are rewriting. Keep the answer focused and "
    "plain — no headers, no fake multiple-choice, no repetition."
)


def _build_user_msg(question: str, draft: str) -> str:
    draft = (draft or "").strip()
    if not draft:
        return f"Question:\n{question}\n\nAnswer:"
    return (
        f"Question:\n{question}\n\n"
        f"Local draft (may be messy; use if useful, ignore if not):\n{draft}\n\n"
        f"Final answer:"
    )


async def refine(
    question: str,
    draft: str,
    *,
    max_tokens: int = 768,
    temperature: float = 0.3,
) -> AsyncIterator[str]:
    streamer = BedrockStreamer()
    async for tok in streamer.stream_text(
        system=_SYSTEM,
        user=_build_user_msg(question, draft),
        max_tokens=max_tokens,
        temperature=temperature,
    ):
        yield tok
