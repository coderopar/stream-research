"""R0.8 full-D0 dry-run primitives."""
from .analysis import (
    DRY_RUN_BOOTSTRAP_DRAWS,
    DRY_RUN_BOOTSTRAP_SEED,
    DRY_RUN_FEATURE_THRESHOLD,
    DRY_RUN_FEATURE_WINDOW_SEC,
    DRY_RUN_HORIZON_SEC,
    DryRunContrast,
    add_mechanical_net_proxy,
    apply_flat_only_position_state_machine,
    attach_future_observability,
    build_causal_analysis_population,
    build_fixed_test_state,
    dependence_aware_contrast,
)

__all__ = [
    "DRY_RUN_BOOTSTRAP_DRAWS", "DRY_RUN_BOOTSTRAP_SEED",
    "DRY_RUN_FEATURE_THRESHOLD", "DRY_RUN_FEATURE_WINDOW_SEC",
    "DRY_RUN_HORIZON_SEC", "DryRunContrast",
    "add_mechanical_net_proxy", "apply_flat_only_position_state_machine",
    "attach_future_observability", "build_causal_analysis_population",
    "build_fixed_test_state", "dependence_aware_contrast",
]
