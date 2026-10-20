"""Exploratory D0/D1/D2 comparison on the immutable 48 historical cases.

Results are developmental; never present them as held-out confirmation.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import yaml

from reviewer_revision.deterministic_retrieval import run_deterministic_retrieval
from reviewer_revision.d2_retrieval import run_d2

ROOT = Path(__file__).resolve().parents[1]
SOURCES = (
    ("primary", ROOT / "agentic_experiments/cases_v1.yaml"),
    ("replication", ROOT / "agentic_experiments/cases_replication_v1.yaml"),
)
OUT = ROOT / "reviewer_revision/results/d2_48case_exploratory"


def main() -> None:
    rows: list[dict] = []
    for study, path in SOURCES:
        cases = yaml.safe_load(path.read_text())["cases"]
        for case in cases:
            d1 = run_deterministic_retrieval(case, max_calls=1)
            d2 = run_d2(case, max_calls=2)
            admissible = case["admissible_actions"]
            rows.append({
                "study": study,
                "case_id": case["id"],
                "stratum": case["stratum"],
                "d0_action": d1["deterministic_before"]["selected_action"],
                "d1_action": d1["selected_action"],
                "d2_action": d2["selected_action"],
                "d0_correct": d1["deterministic_before"]["selected_action"] in admissible,
                "d1_correct": d1["selected_action"] in admissible,
                "d2_correct": d2["selected_action"] in admissible,
                "d1_tool_calls": len(d1["requested_tools"]),
                "d2_tool_calls": len(d2["requested_tools"]),
                "d2_tools": ";".join(d2["requested_tools"]),
                "d2_missing_after": ";".join(d2["missing_fields_after"]),
            })
    assert len(rows) == 48, "Expected frozen 32+16 cases"
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "case_results.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "status": "exploratory_reanalysis_existing_frozen_cases_not_independent_confirmation",
        "n": len(rows),
        "accuracy": {
            arm: {
                "correct": sum(r[arm + "_correct"] for r in rows),
                "total": len(rows),
                "rate": round(sum(r[arm + "_correct"] for r in rows) / len(rows), 6),
            }
            for arm in ("d0", "d1", "d2")
        },
        "tool_calls": {
            arm: sum(r[arm + "_tool_calls"] for r in rows)
            for arm in ("d1", "d2")
        },
        "by_stratum": {
            stratum: {
                "n": len(group),
                "d0_correct": sum(r["d0_correct"] for r in group),
                "d1_correct": sum(r["d1_correct"] for r in group),
                "d2_correct": sum(r["d2_correct"] for r in group),
            }
            for stratum in ("CLEAR", "AMBIGUOUS", "INCOMPLETE", "MISLEADING")
            if (group := [r for r in rows if r["stratum"] == stratum])
        },
        "d1_d2_disagreements": [
            {"study": r["study"], "case_id": r["case_id"],
             "d1_action": r["d1_action"], "d2_action": r["d2_action"]}
            for r in rows if r["d1_action"] != r["d2_action"]
        ],
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
