"""Record of what has been ingested, used for incremental re-ingestion."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


class Manifest:
    """``{source: {content_hash, doc_id, chunk_ids, ingested_at}}`` persisted as JSON.

    ``path=None`` keeps it in memory only (for non-persistent stores).
    """

    def __init__(self, path: str | Path | None) -> None:
        self.path = Path(path) if path is not None else None
        self.entries: dict[str, dict[str, Any]] = {}
        if self.path is not None and self.path.is_file():
            self.entries = json.loads(self.path.read_text(encoding="utf-8"))

    def get(self, source: str) -> dict[str, Any] | None:
        return self.entries.get(source)

    def is_current(self, source: str, content_hash: str) -> bool:
        entry = self.entries.get(source)
        return bool(entry) and entry.get("content_hash") == content_hash  # type: ignore[union-attr]

    def set(self, source: str, **entry: Any) -> None:
        self.entries[source] = entry

    def remove(self, source: str) -> dict[str, Any] | None:
        return self.entries.pop(source, None)

    def clear(self) -> None:
        self.entries = {}

    def save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.entries, indent=1, sort_keys=True), encoding="utf-8")
        os.replace(tmp, self.path)
