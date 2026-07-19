from __future__ import annotations

from pathlib import Path


class RepositoryContextLoader:
    def __init__(self, root: Path | str = ".") -> None:
        self.root = Path(root)

    def load_instructions(self) -> tuple[str, ...]:
        docs = []
        for path in sorted(self.root.glob("**/AGENTS.md")):
            if ".git" in path.parts or ".venv" in path.parts:
                continue
            docs.append(f"# {path.as_posix()}\n" + path.read_text())
        return tuple(docs)
