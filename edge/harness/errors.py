from __future__ import annotations


class HarnessError(RuntimeError):
    """Base error for harness runtime failures."""


class InvalidTriggerError(HarnessError):
    """Raised when a trigger cannot start a run."""


class ResumeError(HarnessError):
    """Raised when a run cannot be resumed from a checkpoint."""
