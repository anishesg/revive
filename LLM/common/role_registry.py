"""Canonical role -> base model mapping. Single source of truth for LLM/."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Role:
    name: str
    base_model: str
    size_label: str
    seq_len: int


_ROLES: tuple[Role, ...] = (
    Role("spotter",     "Qwen/Qwen3-0.6B", "0.6b", 1024),
    Role("drafter",     "Qwen/Qwen3-0.6B", "0.6b", 1024),
    Role("concise",     "Qwen/Qwen3-0.6B", "0.6b", 1024),
    Role("reasoner",    "Qwen/Qwen3-1.7B", "1.7b", 2048),
    Role("writer",      "Qwen/Qwen3-1.7B", "1.7b", 2048),
    Role("critic",      "Qwen/Qwen3-1.7B", "1.7b", 2048),
    Role("factchecker", "Qwen/Qwen3-1.7B", "1.7b", 2048),
    Role("aggregator",  "Qwen/Qwen3-1.7B", "1.7b", 4096),
)

ROLES: dict[str, Role] = {r.name: r for r in _ROLES}
ALL_ROLE_NAMES: tuple[str, ...] = tuple(r.name for r in _ROLES)


def get(role: str) -> Role:
    if role not in ROLES:
        raise KeyError(f"Unknown role: {role!r}. Known: {ALL_ROLE_NAMES}")
    return ROLES[role]


def is_small(role: str) -> bool:
    return get(role).size_label == "0.6b"
