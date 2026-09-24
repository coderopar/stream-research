"""Deterministic interval-overlap components for R0.4 dependence accounting.

Intervals use half-open semantics ``[start_ns, end_ns)``.  Two intervals that
only touch at a boundary are therefore not overlapping.  Connected components
are transitive: if A overlaps B and B overlaps C, all three share a component
whether or not A directly overlaps C.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Iterable, Mapping, Sequence


@dataclass(frozen=True)
class IntervalAssignment:
    component_id: str
    component_size: int
    component_start_ns: int
    component_end_ns: int


def _require_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{field} must be an integer; got {type(value).__name__}")
    return int(value)


def connected_interval_components(
    records: Iterable[Mapping[str, object]],
    *,
    id_field: str = "candidate_id",
    start_field: str = "start_ns",
    end_field: str = "end_ns",
    prefix: str = "OV",
) -> dict[str, IntervalAssignment]:
    """Assign deterministic connected components to valid half-open intervals."""

    rows: list[tuple[int, int, str]] = []
    seen: set[str] = set()
    for record in records:
        key = str(record[id_field])
        if key in seen:
            raise ValueError(f"duplicate interval id: {key}")
        seen.add(key)
        start = _require_int(record[start_field], start_field)
        end = _require_int(record[end_field], end_field)
        if end <= start:
            raise ValueError(f"interval {key} must satisfy end > start")
        rows.append((start, end, key))

    rows.sort(key=lambda x: (x[0], x[1], x[2]))
    if not rows:
        return {}

    components: list[list[tuple[int, int, str]]] = []
    current: list[tuple[int, int, str]] = []
    current_end: int | None = None

    for row in rows:
        start, end, _ = row
        # Half-open semantics: start == current_end merely touches and starts a
        # new component.  start < current_end is true overlap.
        if current and current_end is not None and start >= current_end:
            components.append(current)
            current = []
            current_end = None
        current.append(row)
        current_end = end if current_end is None else max(current_end, end)

    if current:
        components.append(current)

    assignments: dict[str, IntervalAssignment] = {}
    width = max(6, len(str(len(components))))
    for ordinal, members in enumerate(components, start=1):
        component_id = f"{prefix}_{ordinal:0{width}d}"
        cstart = min(x[0] for x in members)
        cend = max(x[1] for x in members)
        size = len(members)
        for _, _, key in members:
            assignments[key] = IntervalAssignment(
                component_id=component_id,
                component_size=size,
                component_start_ns=cstart,
                component_end_ns=cend,
            )
    return assignments


def assert_component_refinement(
    fine: Mapping[str, IntervalAssignment],
    coarse: Mapping[str, IntervalAssignment],
    *,
    keys: Sequence[str] | None = None,
) -> None:
    """Assert every fine component is wholly contained in one coarse component."""

    selected = set(keys) if keys is not None else set(fine) & set(coarse)
    fine_to_coarse: dict[str, str] = {}
    for key in sorted(selected):
        if key not in fine or key not in coarse:
            continue
        f = fine[key].component_id
        c = coarse[key].component_id
        previous = fine_to_coarse.setdefault(f, c)
        if previous != c:
            raise AssertionError(
                f"fine component {f} maps to more than one coarse component: "
                f"{previous}, {c}"
            )
