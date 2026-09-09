from __future__ import annotations

"""Run the 12-case RCAEval adapter-development pilot.

Selection is deterministic: the first two lexicographically sorted case IDs per
fault class. Labels are used only after AAF execution for stratified reporting;
they are never passed to the adapter or AAF.
"""

from pathlib import Path
import json
import sys

import pandas as pd
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import RemoteEntryNotFoundError

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from external_validation.rcaeval.adapter import adapt
from orchestrator.utility import choose_action_details
from orchestrator.cross_domain_v2 import apply_interaction_policy_v2

REPO_ID = "phamquiluan/RCAEval"
OUT = ROOT / "paper_results/external_rcaeval/generated/pilot"
WEIGHTS = (0.45, 0.20, 0.35)


def dl(path: str) -> Path:
    return Path(hf_hub_download(repo_id=REPO_ID, repo_type="dataset", filename=path))


def optional_logs(case: str) -> pd.DataFrame | None:
    try:
        return pd.read_parquet(dl(f"{case}/logs.parquet"))
    except RemoteEntryNotFoundError:
        return None


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    idx = pd.read_parquet(dl("cases.parquet"))
    subset = idx[idx["dataset"].astype(str).str.upper() == "RE2-TT"].copy()

    selected_rows = []
    for fault in sorted(subset["fault"].astype(str).unique()):
        group = subset[subset["fault"].astype(str) == fault].sort_values("case")
        selected_rows.extend([row for _, row in group.head(2).iterrows()])

    records = []
    for row in selected_rows:
        case = str(row["case"])
        metrics = pd.read_parquet(dl(f"{case}/metrics.parquet"))
        logs = optional_logs(case)
        inject_time = int(dl(f"{case}/inject_time.txt").read_text(encoding="utf-8").strip())

        # Adapter receives source telemetry + injection boundary, never labels.
        telemetry, diagnostics = adapt(metrics, inject_time, logs=logs)
        base = choose_action_details(telemetry, WEIGHTS)
        final = apply_interaction_policy_v2(telemetry, str(base["selected_action"]))

        record = {
            "case": case,
            "pilot_role": "adapter-development; excluded from final held-out result",
            "aaf_input": telemetry,
            "adapter_diagnostics": diagnostics,
            "base_action": base["selected_action"],
            "eligible_actions": base["eligible_actions"],
            "selected_action": final["selected_action"],
            "arbitration_reason": final["arbitration_reason"],
            "severities": final["interaction_state"]["severities"],
            "interactions": final["interaction_state"]["interactions"],
            # Audit labels appended only after the decision has been made.
            "audit_fault_label": str(row.get("fault", "")),
            "audit_root_cause_service": str(row.get("root_cause_service", "")),
        }
        records.append(record)
        (OUT / f"{case}.json").write_text(json.dumps(record, indent=2), encoding="utf-8")

    summary = {
        "dataset": "RCAEval RE2-TT",
        "pilot_cases": len(records),
        "selection": "first two lexicographically sorted cases per fault class",
        "label_blind_decision_path": True,
        "weights": list(WEIGHTS),
        "adapter_policy": "decision-active latency/CPU; memory/disk/socket/logs retained as auxiliary measured evidence",
        "action_counts": pd.Series([r["selected_action"] for r in records]).value_counts().to_dict(),
        "by_fault": {},
        "important": "Pilot is for adapter validation only and is excluded from the final held-out evaluation.",
    }
    for fault in sorted({r["audit_fault_label"] for r in records}):
        rows = [r for r in records if r["audit_fault_label"] == fault]
        summary["by_fault"][fault] = {
            "cases": len(rows),
            "actions": [r["selected_action"] for r in rows],
            "reliability_severity": [r["severities"]["reliability"] for r in rows],
        }

    (OUT / "pilot_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pd.DataFrame([{
        "case": r["case"],
        "fault": r["audit_fault_label"],
        "root_cause_service": r["audit_root_cause_service"],
        "reliability_severity": r["severities"]["reliability"],
        "selected_action": r["selected_action"],
        "arbitration_reason": r["arbitration_reason"],
        "logs_available": r["adapter_diagnostics"]["auxiliary_evidence"]["logs"].get("available", False),
    } for r in records]).to_csv(OUT / "pilot_case_results.csv", index=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
