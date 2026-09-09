from __future__ import annotations

"""Execute the frozen 78-case RCAEval RE2-TT held-out evaluation."""

from pathlib import Path
import hashlib
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
OUT = ROOT / "paper_results/external_rcaeval/generated/heldout"
WEIGHTS = (0.45, 0.20, 0.35)


def dl(path: str) -> Path:
    return Path(hf_hub_download(repo_id=REPO_ID, repo_type="dataset", filename=path))


def optional_logs(case: str) -> pd.DataFrame | None:
    try:
        return pd.read_parquet(dl(f"{case}/logs.parquet"))
    except RemoteEntryNotFoundError:
        return None


def pilot_case_ids(subset: pd.DataFrame) -> set[str]:
    selected: set[str] = set()
    for fault in sorted(subset["fault"].astype(str).unique()):
        group = subset[subset["fault"].astype(str) == fault].sort_values("case")
        selected.update(group.head(2)["case"].astype(str).tolist())
    return selected


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    idx = pd.read_parquet(dl("cases.parquet"))
    subset = idx[idx["dataset"].astype(str).str.upper() == "RE2-TT"].copy()
    excluded = pilot_case_ids(subset)
    heldout = subset[~subset["case"].astype(str).isin(excluded)].sort_values("case")
    if len(subset) != 90 or len(excluded) != 12 or len(heldout) != 78:
        raise RuntimeError(f"Frozen split violated: total={len(subset)} pilot={len(excluded)} heldout={len(heldout)}")

    records = []
    for _, row in heldout.iterrows():
        case = str(row["case"])
        metrics = pd.read_parquet(dl(f"{case}/metrics.parquet"))
        logs = optional_logs(case)
        inject_time = int(dl(f"{case}/inject_time.txt").read_text(encoding="utf-8").strip())

        telemetry, diagnostics = adapt(metrics, inject_time, logs=logs)
        base = choose_action_details(telemetry, WEIGHTS)
        final = apply_interaction_policy_v2(telemetry, str(base["selected_action"]))

        record = {
            "case": case,
            "evaluation_role": "held-out; not used for adapter development",
            "aaf_input": telemetry,
            "adapter_diagnostics": diagnostics,
            "base_action": base["selected_action"],
            "eligible_actions": base["eligible_actions"],
            "selected_action": final["selected_action"],
            "arbitration_reason": final["arbitration_reason"],
            "severities": final["interaction_state"]["severities"],
            "interactions": final["interaction_state"]["interactions"],
            # Labels appended after decision for stratified audit/reporting only.
            "audit_fault_label": str(row.get("fault", "")),
            "audit_root_cause_service": str(row.get("root_cause_service", "")),
        }
        records.append(record)
        (OUT / f"{case}.json").write_text(json.dumps(record, indent=2), encoding="utf-8")

    flat = pd.DataFrame([{
        "case": r["case"],
        "fault": r["audit_fault_label"],
        "root_cause_service": r["audit_root_cause_service"],
        "reliability_severity": r["severities"]["reliability"],
        "selected_action": r["selected_action"],
        "arbitration_reason": r["arbitration_reason"],
        "latency_proxy_active": "p95_latency_ms" in r["aaf_input"]["sre"],
        "cpu_saturation_active": "saturation_pct" in r["aaf_input"]["sre"],
        "logs_available": r["adapter_diagnostics"]["auxiliary_evidence"]["logs"].get("available", False),
    } for r in records])
    csv_path = OUT / "heldout_case_results.csv"
    flat.to_csv(csv_path, index=False)

    by_fault = {}
    for fault, group in flat.groupby("fault"):
        by_fault[str(fault)] = {
            "cases": int(len(group)),
            "action_counts": group["selected_action"].value_counts().to_dict(),
            "mean_reliability_severity": float(group["reliability_severity"].mean()),
            "latency_proxy_active": int(group["latency_proxy_active"].sum()),
            "cpu_saturation_active": int(group["cpu_saturation_active"].sum()),
        }

    summary = {
        "dataset": "RCAEval RE2-TT",
        "total_dataset_cases": 90,
        "excluded_adapter_development_cases": 12,
        "heldout_cases": len(records),
        "label_blind_decision_path": True,
        "weights": list(WEIGHTS),
        "action_counts": flat["selected_action"].value_counts().to_dict(),
        "material_action_cases": int((flat["selected_action"] != "No action (observe)").sum()),
        "observe_cases": int((flat["selected_action"] == "No action (observe)").sum()),
        "by_fault": by_fault,
        "case_csv_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
        "interpretation_guardrail": "RCAEval fault labels are not governance-action labels; action accuracy is not reported.",
    }
    (OUT / "heldout_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
