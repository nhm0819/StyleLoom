"""Long-term memory: the style schemas.

A `style.json` is the reusable asset of this system -- extracted once from
reference videos, then read by every run. It is stored as a plain file because
it is meant to be opened, read and hand-corrected when the extractor mislabels
something. The schema is the contract, not the extractor.
"""

from __future__ import annotations

from pathlib import Path

from ..config import Settings
from ..errors import NotFoundError
from ..schema import StyleSchema


class StyleStore:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def dir_for(self, style_id: str) -> Path:
        return self.settings.styles_dir / style_id

    def path_for(self, style_id: str) -> Path:
        return self.dir_for(style_id) / "style.json"

    def exists(self, style_id: str) -> bool:
        return self.path_for(style_id).exists()

    def save(self, style: StyleSchema) -> Path:
        path = self.path_for(style.style_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(style.model_dump_json(indent=2), encoding="utf-8")
        return path

    def load(self, style_id: str) -> StyleSchema:
        path = self.path_for(style_id)
        if not path.exists():
            raise NotFoundError(
                f"style not found: {style_id!r}. "
                f"Extract one first, or check {self.settings.styles_dir}."
            )
        return StyleSchema.model_validate_json(path.read_text(encoding="utf-8"))

    def list_ids(self) -> list[str]:
        root = self.settings.styles_dir
        if not root.exists():
            return []
        return sorted(p.name for p in root.iterdir() if (p / "style.json").exists())
