from __future__ import annotations

import unittest

from edge.harness.events import EventType, HarnessEvent
from edge.harness.logging import REDACTED, StructuredEventLogger, redact
from edge.loop.state import LoopState, validate_transition


class RedactionAndStateTests(unittest.TestCase):
    def test_secret_redaction(self) -> None:
        payload = {"api_key": "secret", "nested": {"token": "abc", "safe": "value"}}
        redacted = redact(payload)

        self.assertEqual(redacted["api_key"], REDACTED)
        self.assertEqual(redacted["nested"]["token"], REDACTED)
        self.assertEqual(redacted["nested"]["safe"], "value")

    def test_event_logger_redacts_details(self) -> None:
        logger = StructuredEventLogger()
        logger.emit(HarnessEvent(EventType.RUN_STARTED, run_id="run", details={"api_key": "secret"}))

        self.assertEqual(logger.events[0]["details"]["api_key"], REDACTED)

    def test_state_transition_validity(self) -> None:
        validate_transition(LoopState.CREATED, LoopState.CONTEXT_READY)
        with self.assertRaises(ValueError):
            validate_transition(LoopState.CREATED, LoopState.COMPLETED)


if __name__ == "__main__":
    unittest.main()
