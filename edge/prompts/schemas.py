from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PromptMessage:
    role: str
    content: str


@dataclass(frozen=True)
class PromptBundle:
    version: str
    messages: tuple[PromptMessage, ...]
