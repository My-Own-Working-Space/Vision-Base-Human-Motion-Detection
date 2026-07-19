from __future__ import annotations

from edge.loop.actions import Action


class ActionSelectionPolicy:
    def select_next(self, plan: tuple[Action, ...]) -> Action:
        if not plan:
            raise ValueError("Plan is empty")
        return plan[0]
