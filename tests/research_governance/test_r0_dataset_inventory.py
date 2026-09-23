from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "research"
    / "r0"
    / "provenance"
    / "build_dataset_inventory.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("r0_inventory", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_date_partition_and_cohort_classification():
    m = load_module()
    assert m.parse_file_date(Path("perps/trades/date=2026-06-01/part.parquet")).isoformat() == "2026-06-01"
    assert m.classify_cohort(m.date(2026, 5, 17)) == "PRE_D0"
    assert m.classify_cohort(m.date(2026, 5, 18)) == "D0"
    assert m.classify_cohort(m.date(2026, 7, 18)) == "D0"
    assert m.classify_cohort(m.date(2026, 7, 19)) == "P1_PROTECTED"
    assert m.classify_cohort(None) == "UNDATED"
    assert m.classify_record(Path("_extraction_state/schema_manifest.json"), None) == (
        "CANONICAL_METADATA",
        "NOT_APPLICABLE",
    )
    assert m.classify_record(Path("mystery.bin"), None) == (
        "UNKNOWN_UNDATED",
        "UNDATED",
    )


def test_feed_key_stops_at_date_partition():
    m = load_module()
    rel = Path("perps/depth_updates/date=2026-07-19/part-000.parquet")
    assert m.feed_key_for(rel) == "perps/depth_updates"


def test_scanner_refuses_noncanonical_tree(tmp_path: Path):
    m = load_module()
    repo = tmp_path / "repo"
    canonical = repo / "data" / "derived" / "outcomes"
    canonical.mkdir(parents=True)
    out = repo / "research" / "governance" / "datasets"
    with pytest.raises(SystemExit, match="REFUSED"):
        m.validate_roots(repo, canonical, out)


def test_scanner_accepts_canonical_tree(tmp_path: Path):
    m = load_module()
    repo = tmp_path / "repo"
    canonical = repo / "data" / "canonical" / "market_events_v1"
    canonical.mkdir(parents=True)
    out = repo / "research" / "governance" / "datasets"
    physical = m.validate_roots(repo, canonical, out)
    assert physical == canonical.resolve()


def test_scanner_accepts_approved_external_root_symlink_when_supported(tmp_path: Path):
    m = load_module()
    repo = tmp_path / "repo"
    canonical_parent = repo / "data" / "canonical"
    canonical_parent.mkdir(parents=True)
    external = tmp_path / "external_canonical"
    external.mkdir()
    (external / "sample.bin").write_bytes(b"x")
    logical = canonical_parent / "market_events_v1"
    try:
        logical.symlink_to(external, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("directory symlinks unavailable on this platform")
    out = repo / "research" / "governance" / "datasets"
    physical = m.validate_roots(repo, logical, out)
    assert physical == external.resolve()
    assert m.enumerate_files(physical) == [external.resolve() / "sample.bin"]


def test_symlink_escape_is_rejected_when_supported(tmp_path: Path):
    m = load_module()
    root = tmp_path / "canonical"
    root.mkdir()
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"x")
    link = root / "escaped.bin"
    try:
        link.symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable on this platform")
    with pytest.raises(RuntimeError, match="Nested symlink is not allowed"):
        m.enumerate_files(root)


def test_topology_detects_external_storage_by_resolved_path_difference(tmp_path: Path):
    m = load_module()
    repo = tmp_path / "repo"
    canonical_parent = repo / "data" / "canonical"
    canonical_parent.mkdir(parents=True)
    external = tmp_path / "external_canonical"
    external.mkdir()
    logical = canonical_parent / "market_events_v1"
    try:
        logical.symlink_to(external, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("directory symlinks unavailable on this platform")
    out = repo / "research" / "governance" / "datasets"
    physical = m.validate_roots(repo, logical, out)
    assert physical != logical.absolute()
    assert physical == external.resolve()
