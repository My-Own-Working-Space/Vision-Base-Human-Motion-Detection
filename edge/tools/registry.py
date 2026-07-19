from __future__ import annotations

from edge.tools.base import Tool


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.metadata.name] = tool

    def get(self, name: str) -> Tool:
        return self._tools[name]

    def list_names(self) -> list[str]:
        return sorted(self._tools)
