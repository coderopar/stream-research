"""R0.4 dependence and episode primitives."""

from .episodes import (
    EpisodeAssignment,
    conservative_day_overlap_clusters,
    exact_lineage_groups,
    gap_connected_episodes,
    partition_episodes_by_value,
)
from .intervals import (
    IntervalAssignment,
    assert_component_refinement,
    connected_interval_components,
)

__all__ = [
    "EpisodeAssignment",
    "IntervalAssignment",
    "assert_component_refinement",
    "connected_interval_components",
    "conservative_day_overlap_clusters",
    "exact_lineage_groups",
    "gap_connected_episodes",
    "partition_episodes_by_value",
]
