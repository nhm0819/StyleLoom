"""Episodic memory: which pool entries a style has used recently.

Feeds the recency penalty in the hook and casting tools, which is what stops a
batch of three videos from opening the same way, with the same presenter, in the
same room, three times.

Stored as an append-only JSONL log per style rather than derived by scanning every
run: the previous implementation loaded and parsed every `run.json` on disk to find
the last four archetypes, which is O(all runs) on every run. A tail read of one
small file is O(window), and the log is still a text file anyone can open.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..config import Settings
from ..schema import Choice

FILENAME = "choices.jsonl"
# Tail bytes to read. Comfortably covers a few dozen lines across all kinds, which
# is far more than any recency window needs.
_TAIL_BYTES = 8192


class ChoiceHistory:
    """Recent pool selections per style, filtered by kind."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def path_for(self, style_id: str) -> Path:
        return self.settings.styles_dir / style_id / FILENAME

    def append(self, style_id: str, choice: Choice) -> None:
        path = self.path_for(style_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(choice.model_dump_json() + "\n")

    def _read_tail(self, style_id: str) -> list[Choice]:
        path = self.path_for(style_id)
        if not path.exists():
            return []
        with path.open("rb") as fh:
            fh.seek(0, 2)
            size = fh.tell()
            fh.seek(max(size - _TAIL_BYTES, 0))
            blob = fh.read()
        # Seeking into the middle of a file can land mid-line, and a crashed write
        # can leave a partial record. Both fail to parse and are skipped.
        out: list[Choice] = []
        for line in reversed(blob.decode("utf-8", errors="ignore").splitlines()):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(Choice.model_validate_json(line))
            except (ValueError, json.JSONDecodeError):
                continue
        return out

    def recent(
        self, style_id: str, kind: str, limit: int | None = None
    ) -> list[Choice]:
        """Most recent first."""
        n = self.settings.hook_recency_window if limit is None else limit
        if n <= 0:
            return []
        matching = [c for c in self._read_tail(style_id) if c.kind == kind]
        return matching[:n]

    def recent_values(
        self, style_id: str, kind: str, limit: int | None = None
    ) -> list[str]:
        return [c.value for c in self.recent(style_id, kind, limit)]

    def all_recent(self, style_id: str, limit: int = 30) -> list[Choice]:
        """Every kind, interleaved, most recent first. For inspection commands."""
        return self._read_tail(style_id)[:limit]
