from __future__ import annotations

from edge.context.models import IterationContext
from edge.prompts.schemas import PromptBundle, PromptMessage
from edge.prompts.templates import PROMPT_VERSION, SYSTEM_TEMPLATE


class PromptBuilder:
    version = PROMPT_VERSION

    def build(self, context: IterationContext) -> PromptBundle:
        return PromptBundle(
            version=self.version,
            messages=(
                PromptMessage(role="system", content=SYSTEM_TEMPLATE),
                PromptMessage(role="user", content=f"Goal: {context.goal}"),
            ),
        )
