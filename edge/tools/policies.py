from __future__ import annotations

from enum import Enum


class SideEffect(str, Enum):
    NONE = "none"
    READ_ONLY = "read_only"
    WRITES_LOCAL = "writes_local"
    EXTERNAL_CALL = "external_call"


class RetrySafety(str, Enum):
    SAFE = "safe"
    UNSAFE = "unsafe"


class Idempotency(str, Enum):
    IDEMPOTENT = "idempotent"
    NON_IDEMPOTENT = "non_idempotent"
