"""Append-only JSONL registry storage."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable

from .canonical import canonical_json_bytes
from .models import RegistryValidationError


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_no, raw in enumerate(fh, 1):
            text = raw.strip()
            if not text:
                continue
            try:
                value = json.loads(text)
            except json.JSONDecodeError as exc:
                raise RegistryValidationError(f"invalid JSONL at {path}:{line_no}") from exc
            if not isinstance(value, dict):
                raise RegistryValidationError(f"registry row must be a JSON object at {path}:{line_no}")
            records.append(value)
    return records


def append_jsonl(
    path: Path,
    record: dict[str, Any],
    *,
    unique_key: Callable[[dict[str, Any]], Any] | None = None,
) -> None:
    """Append exactly one canonical JSON record after duplicate-key validation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if unique_key is not None:
        key = unique_key(record)
        for existing in read_jsonl(path):
            if unique_key(existing) == key:
                raise RegistryValidationError(f"duplicate registry key: {key!r}")
    payload = canonical_json_bytes(record) + b"\n"
    with path.open("ab") as fh:
        fh.write(payload)
        fh.flush()
        os.fsync(fh.fileno())
