"""Canonical hashing and reproducibility primitives for governed research artifacts."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from typing import Any


class ReproducibilityError(ValueError):
    """Raised when a supposedly immutable/substantive artifact changes."""


def canonical_json_bytes(value: Any) -> bytes:
    """Return a deterministic UTF-8 JSON encoding suitable for hashing."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sha256_canonical(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def with_freeze_hash(record: Mapping[str, Any]) -> dict[str, Any]:
    """Return a copy with freeze_hash computed over every other field."""
    payload = dict(record)
    payload.pop("freeze_hash", None)
    payload["freeze_hash"] = sha256_canonical(payload)
    return payload


def verify_freeze_hash(record: Mapping[str, Any]) -> None:
    expected = record.get("freeze_hash")
    if not isinstance(expected, str) or len(expected) != 64:
        raise ReproducibilityError("record is missing a valid freeze_hash")
    payload = dict(record)
    payload.pop("freeze_hash", None)
    actual = sha256_canonical(payload)
    if actual != expected:
        raise ReproducibilityError(f"freeze_hash mismatch: expected={expected} actual={actual}")


def substantive_hash(value: Mapping[str, Any], *, volatile_fields: Iterable[str] = ()) -> str:
    """Hash substantive content while explicitly excluding named volatile metadata.

    This is the R0.7 idempotence primitive. Volatile fields must be named explicitly;
    there is no wildcard exclusion.
    """
    volatile = set(volatile_fields)
    payload = {k: v for k, v in value.items() if k not in volatile}
    return sha256_canonical(payload)


def assert_substantive_equal(
    previous: Mapping[str, Any],
    current: Mapping[str, Any],
    *,
    volatile_fields: Iterable[str] = (),
) -> None:
    a = substantive_hash(previous, volatile_fields=volatile_fields)
    b = substantive_hash(current, volatile_fields=volatile_fields)
    if a != b:
        raise ReproducibilityError(f"substantive artifact changed: {a} != {b}")


def assert_append_only_prefix(previous_bytes: bytes, current_bytes: bytes) -> None:
    """Verify an append-only registry has not rewritten or deleted prior bytes."""
    if not current_bytes.startswith(previous_bytes):
        raise ReproducibilityError("append-only registry prefix was modified or deleted")
