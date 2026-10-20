"""Run D1 on the immutable 32+16 cases. New exploratory outputs only."""
from __future__ import annotations
import csv
import json
from pathlib import Path
import yaml
from reviewer_revision.deterministic_retrieval import run_deterministic_retrieval

ROOT = Path(__file__).resolve().parents[1]
SOURCES = [
    ("primary", ROOT / "agentic_experiments/cases_v1.yaml"),
    ("replication", ROOT / "agentic_experiments/cases_replication_v1.yaml"),
]
OUT = ROOT / "reviewer_revision/results/d1_48case_exploratory"

def main() -> None:
    rows = []
    for study, path in SOURCES:
        for case in yaml.safe_load(path.read_text())["cases"]:
            result = run_deterministic_retrieval(case)
            rows.append({
                "study": study, "case_id": case["id"], "stratum": case["stratum"],
                "before_action": result["deterministic_before"]["selected_action"],
                "after_action": result["selected_action"],
                "before_correct": result["deterministic_before"]["selected_action"] in case["admissible_actions"],
                "after_correct": result["selected_action"] in case["admissible_actions"],
                "requested_tools": ";".join(result["requested_tools"]),
                "missing_before": ";".join(result["missing_fields_before"]),
                "missing_after": ";".join(result["missing_fields_after"]),
            })
    assert len(rows) == 48
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "case_results.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "status": "exploratory_reanalysis_existing_frozen_cases",
        "n": len(rows),
        "before_correct": sum(r["before_correct"] for r in rows),
        "after_correct": sum(r["after_correct"] for r in rows),
        "by_study_stratum": {
            study + "/" + stratum: {
                "n": len(rs),
                "before_correct": sum(r["before_correct"] for r in rs),
                "after_correct": sum(r["after_correct"] for r in rs),
                "tools_requested": sum(bool(r["requested_tools"]) for r in rs),
            }
            for study in ("primary", "replication")
            for stratum in ("CLEAR", "AMBIGUOUS", "INCOMPLETE", "MISLEADING")
            if (rs := [r for r in rows if r["study"] == study and r["stratum"] == stratum])
        },
        "recovered_case_details": [r for r in rows if r["case_id"] in ("AG-I01", "RP-I01")],
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
