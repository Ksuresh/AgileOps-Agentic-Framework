"""Produce reproducible RCAEval case-level eligibility and leakage manifest.

Uses the already downloaded public index; this does not imply raw telemetry access.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent / "results/external_dataset_audit"

def main():
    source = ROOT / "rcaeval_cases.parquet"
    if not source.exists():
        raise FileNotFoundError("Run audit_rcaeval_index first")
    df = pd.read_parquet(source)
    required = ["case", "suite", "system", "fault", "root_cause_service", "has_logs", "has_traces"]
    missing = set(required) - set(df)
    if missing:
        raise ValueError("Missing required columns: " + repr(sorted(missing)))
    if df["case"].duplicated().any():
        raise ValueError("Duplicate case IDs")
    rows = []
    for record in df.to_dict(orient="records"):
        group_parts = [str(record[k]) for k in ("system", "fault", "root_cause_service")]
        group = hashlib.sha256(json.dumps(group_parts).encode()).hexdigest()[:16]
        rows.append({
            "case": str(record["case"]),
            "suite": str(record["suite"]),
            "system": str(record["system"]),
            "fault": str(record["fault"]),
            "leakage_family_id": group,
            "has_logs": bool(record["has_logs"]),
            "has_traces": bool(record["has_traces"]),
            "eligible_for_telemetry_review": bool(record["has_logs"] or record["has_traces"]),
            "raw_telemetry_verified": False,
            "governance_label_verified": False,
        })
    result = pd.DataFrame(rows).sort_values("case")
    result.to_csv(ROOT / "rcaeval_case_eligibility_manifest.csv", index=False)
    summary = {
        "status": "index_only_eligibility_not_confirmed_incident_cohort",
        "case_count": len(result),
        "telemetry_flagged_cases": int(result["eligible_for_telemetry_review"].sum()),
        "metrics_only_cases": int((~result["eligible_for_telemetry_review"]).sum()),
        "unique_leakage_families": int(result["leakage_family_id"].nunique()),
        "raw_telemetry_verified_cases": 0,
        "governance_labeled_cases": 0,
        "note": "Eligibility is a candidate screening flag only. No train/test split or governance ground truth is inferred.",
    }
    (ROOT / "rcaeval_eligibility_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
