"""Outcome-blind episode constructions for R0.4."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Iterable, Mapping

from .intervals import IntervalAssignment


@dataclass(frozen=True)
class EpisodeAssignment:
    episode_id: str
    episode_size: int
    episode_start_ns: int
    episode_end_ns: int


def _require_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{field} must be an integer; got {type(value).__name__}")
    return int(value)


def gap_connected_episodes(
    records: Iterable[Mapping[str, object]],
    *,
    max_gap_ns: int,
    id_field: str = "candidate_id",
    time_field: str = "candidate_time_ns",
    prefix: str = "EP",
) -> dict[str, EpisodeAssignment]:
    """Connect consecutive candidate timestamps when the gap is <= max_gap_ns."""

    max_gap = _require_int(max_gap_ns, "max_gap_ns")
    if max_gap < 0:
        raise ValueError("max_gap_ns must be >= 0")

    rows: list[tuple[int, str]] = []
    seen: set[str] = set()
    for record in records:
        key = str(record[id_field])
        if key in seen:
            raise ValueError(f"duplicate episode id: {key}")
        seen.add(key)
        rows.append((_require_int(record[time_field], time_field), key))
    rows.sort(key=lambda x: (x[0], x[1]))
    if not rows:
        return {}

    groups: list[list[tuple[int, str]]] = []
    current: list[tuple[int, str]] = []
    previous_time: int | None = None
    for time_ns, key in rows:
        if current and previous_time is not None and time_ns - previous_time > max_gap:
            groups.append(current)
            current = []
        current.append((time_ns, key))
        previous_time = time_ns
    if current:
        groups.append(current)

    result: dict[str, EpisodeAssignment] = {}
    width = max(6, len(str(len(groups))))
    for ordinal, members in enumerate(groups, start=1):
        episode_id = f"{prefix}_{ordinal:0{width}d}"
        start_ns = min(x[0] for x in members)
        end_ns = max(x[0] for x in members)
        size = len(members)
        for _, key in members:
            result[key] = EpisodeAssignment(episode_id, size, start_ns, end_ns)
    return result


def partition_episodes_by_value(
    base: Mapping[str, EpisodeAssignment],
    value_by_id: Mapping[str, object],
    *,
    prefix: str = "DEP",
) -> dict[str, EpisodeAssignment]:
    """Partition each base episode by an exact categorical value.

    This is deliberately not a global contiguous-run algorithm: simultaneous
    opposite-direction candidates do not get an arbitrary order.  Within a
    temporal episode, all candidates sharing the same value form one exact
    subepisode.
    """

    groups: dict[tuple[str, str], list[str]] = {}
    for key, assignment in base.items():
        if key not in value_by_id:
            raise KeyError(f"missing partition value for {key}")
        group_key = (assignment.episode_id, repr(value_by_id[key]))
        groups.setdefault(group_key, []).append(key)

    ordered_groups = sorted(
        groups.items(),
        key=lambda kv: (
            min(base[k].episode_start_ns for k in kv[1]),
            kv[0][0],
            kv[0][1],
            min(kv[1]),
        ),
    )

    result: dict[str, EpisodeAssignment] = {}
    width = max(6, len(str(len(ordered_groups))))
    for ordinal, (_, members) in enumerate(ordered_groups, start=1):
        episode_id = f"{prefix}_{ordinal:0{width}d}"
        start_ns = min(base[k].episode_start_ns for k in members)
        end_ns = max(base[k].episode_end_ns for k in members)
        size = len(members)
        for key in members:
            result[key] = EpisodeAssignment(episode_id, size, start_ns, end_ns)
    return result


def exact_lineage_groups(
    records: Iterable[Mapping[str, object]],
    *,
    id_field: str = "candidate_id",
    group_fields: tuple[str, ...],
    time_field: str = "candidate_time_ns",
    prefix: str = "LIN",
) -> dict[str, EpisodeAssignment]:
    """Group candidates by an exact, already-recorded lineage tuple."""

    grouped: dict[tuple[object, ...], list[tuple[int, str]]] = {}
    seen: set[str] = set()
    for record in records:
        key = str(record[id_field])
        if key in seen:
            raise ValueError(f"duplicate lineage id: {key}")
        seen.add(key)
        group_key = tuple(record[field] for field in group_fields)
        grouped.setdefault(group_key, []).append(
            (_require_int(record[time_field], time_field), key)
        )

    ordered = sorted(
        grouped.items(),
        key=lambda kv: (min(x[0] for x in kv[1]), repr(kv[0]), min(x[1] for x in kv[1])),
    )
    result: dict[str, EpisodeAssignment] = {}
    width = max(6, len(str(len(ordered))))
    for ordinal, (_, members) in enumerate(ordered, start=1):
        episode_id = f"{prefix}_{ordinal:0{width}d}"
        start_ns = min(x[0] for x in members)
        end_ns = max(x[0] for x in members)
        size = len(members)
        for _, key in members:
            result[key] = EpisodeAssignment(episode_id, size, start_ns, end_ns)
    return result


class _DisjointSet:
    def __init__(self, values: Iterable[str]) -> None:
        self.parent = {v: v for v in values}

    def find(self, x: str) -> str:
        parent = self.parent[x]
        if parent != x:
            self.parent[x] = self.find(parent)
        return self.parent[x]

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        # Stable representative independent of input ordering.
        lo, hi = sorted((ra, rb))
        self.parent[hi] = lo


def conservative_day_overlap_clusters(
    day_by_id: Mapping[str, str],
    overlap_30m: Mapping[str, IntervalAssignment],
    *,
    prefix: str = "DAY30",
) -> dict[str, str]:
    """Return the union of UTC-day clustering and 30m overlap clustering.

    Every candidate on the same UTC candidate date shares a cluster.  If a 30m
    overlap component spans multiple dates, those day clusters are unioned so a
    cross-midnight dependent path cannot be split by the calendar boundary.
    """

    days = sorted(set(str(v) for v in day_by_id.values()))
    dsu = _DisjointSet(days)
    overlap_days: dict[str, set[str]] = {}
    for key, assignment in overlap_30m.items():
        if key not in day_by_id:
            raise KeyError(f"missing candidate day for {key}")
        overlap_days.setdefault(assignment.component_id, set()).add(str(day_by_id[key]))
    for ds in overlap_days.values():
        ordered = sorted(ds)
        for day in ordered[1:]:
            dsu.union(ordered[0], day)

    roots = sorted({dsu.find(day) for day in days})
    root_id = {root: f"{prefix}_{i:06d}" for i, root in enumerate(roots, start=1)}
    return {key: root_id[dsu.find(str(day))] for key, day in day_by_id.items()}
