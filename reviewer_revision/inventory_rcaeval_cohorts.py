"""Pre-outcome RCAEval cohort inventory and leakage-group audit.

Only index metadata is used. No governance labels or model predictions are read.
This script does NOT choose a held-out set; split design requires preregistration.
"""
from __future__ import annotations

import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "reviewer_revision/results/external_dataset_audit/rcaeval_cases.parquet"
OUT = ROOT / "reviewer_revision/results/external_dataset_audit/rcaeval_cohort_inventory.json"

def main() -> None:
    if not SOURCE.exists():
        raise FileNotFoundError("Run audit_rcaeval_index first to acquire the source index")
    frame = pd.read_parquet(SOURCE)
    for field in ("dataset", "suite", "system", "fault", "root_cause_service", "repetition", "case"):
        if field not in frame.columns:
            raise ValueError("Required metadata missing: " + field)
    if frame["case"].duplicated().any():
        raise ValueError("Duplicate case IDs")
    grouped = frame.groupby(["suite", "system"], dropna=False)
    cohorts = []
    for (suite, system), group in grouped:
        cohorts.append({
            "suite": str(suite), "system": str(system),
            "n": int(len(group)),
            "with_logs": int(group["has_logs"].sum()),
            "with_traces": int(group["has_traces"].sum()),
            "with_both": int((group["has_logs"] & group["has_traces"]).sum()),
            "fault_types": int(group["fault"].nunique()),
            "root_cause_services": int(group["root_cause_service"].nunique()),
        })
    # A conservative candidate grouping: all repetitions of the same
    # system + fault type + root-cause service remain in the same fold.
    group_columns = ["system", "fault", "root_cause_service"]
    family_sizes = frame.groupby(group_columns, dropna=False).size()
    report = {
        "status": "pre_outcome_cohort_inventory_not_split_not_governance_labels",
        "total_cases": int(len(frame)),
        "cohorts": cohorts,
        "candidate_family_grouping": group_columns,
        "candidate_families": int(len(family_sizes)),
        "largest_family": int(family_sizes.max()),
        "families_with_repeated_cases": int((family_sizes > 1).sum()),
        "caution": "Do not randomly split repeated fault families; finalize cross-system or grouped split and expert-labeling plan before evaluation.",
    }
    OUT.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
