from __future__ import annotations

from benchmark.reviewer_oracle_specs import REVIEWER_ORACLE_SPECS
from scenario_generator.reviewer_aligned import generate_reviewer_aligned_scenarios


def test_reviewer_aligned_suite_has_54_unique_cases():
    cases = generate_reviewer_aligned_scenarios(seed=42)
    ids = [c["scenario_id"] for c in cases]
    assert len(cases) == 54
    assert len(set(ids)) == 54
    assert ids[0] == "TC-01"
    assert ids[-1] == "TC-54"


def test_every_case_has_frozen_oracle():
    cases = generate_reviewer_aligned_scenarios(seed=42)
    missing = [c["scenario_id"] for c in cases if c["scenario_id"] not in REVIEWER_ORACLE_SPECS]
    assert missing == []


def test_original_30_are_preserved_as_prefix():
    cases = generate_reviewer_aligned_scenarios(seed=42)
    assert [c["scenario_id"] for c in cases[:30]] == [f"TC-{i:02d}" for i in range(1, 31)]


def test_new_group_counts_match_frozen_design():
    cases = generate_reviewer_aligned_scenarios(seed=42)
    added = cases[30:]
    counts = {}
    for case in added:
        counts[case["experiment_group"]] = counts.get(case["experiment_group"], 0) + 1
    assert counts == {"G5": 6, "G4": 8, "G3": 10}


def test_seed_variants_do_not_change_oracles_or_groups():
    a = generate_reviewer_aligned_scenarios(seed=42)
    b = generate_reviewer_aligned_scenarios(seed=46)
    for x, y in zip(a, b):
        assert x["scenario_id"] == y["scenario_id"]
        assert x["experiment_group"] == y["experiment_group"]
        assert x["ground_truth"] == y["ground_truth"]


def test_explicit_missing_cases_use_provenance_schema():
    by_id = {c["scenario_id"]: c for c in generate_reviewer_aligned_scenarios(seed=42)}
    assert by_id["TC-37"]["telemetry"]["deploy"]["_evidence"]["config_drift"]["status"] == "missing"
    assert by_id["TC-38"]["telemetry"]["sre"]["_evidence"]["p95_latency_ms"]["status"] == "missing"
    assert by_id["TC-39"]["telemetry"]["finops"]["_evidence"]["cost_spike_pct"]["status"] == "missing"
    assert by_id["TC-40"]["telemetry"]["sec"]["_evidence"]["critical_cves"]["status"] == "missing"
