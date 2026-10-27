"""R4-1 / R5-1 / R6-1: deterministic auto-fetch baseline on the frozen 32 + 16 selective-AEI cases. No LLM.

B1  deterministic AAF (no tools)
B2a reviewer's literal rule: for every missing evidence field that maps to an approved read-only tool available in
    the case, call that tool once and re-run deterministic governance (no trigger)
B2b same fetch rule, but only when Hybrid's selective trigger fires (isolates the LLM's tool choice from the trigger)
The field->tool map follows the tool-family semantics of agentic_experiments/FROZEN_PROTOCOL.md; it does not read
the evaluator-only `expected_tool`. Tool execution uses the same frozen rule as hybrid.py (a tool returns the case's
recorded result only if it is the case's evidence-bearing tool; otherwise "no additional evidence").
"""
from __future__ import annotations

import copy
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agentic_experiments.agentic_runtime import merge_tool_result  # noqa: E402
from agentic_experiments.hybrid import deterministic_assessment, missing_fields, selective_trigger  # noqa: E402

FIELD_TO_TOOL = {
    "deploy.restart_burst_count": "get_restart_history", "deploy.restart_window_seconds": "get_restart_history",
    "deploy.config_drift": "get_deployment_history", "deploy.pipeline_failed": "get_pipeline_history",
    "deploy.artifact_mismatch": "get_deployment_history",
    "sec.policy_violation": "get_policy_gate_status", "sec.critical_cves": "get_vulnerability_details",
    "finops.hpa_scale_to": "get_replica_history", "finops.cost_spike_pct": "get_cost_proxy_history",
    "sre.error_rate_pct": "get_error_rate_history", "sre.p95_latency_ms": "get_latency_history",
    "sre.saturation_pct": "get_saturation_history",
}
OUT = Path(__file__).resolve().parent / "results"


def fetch_and_decide(case: dict, gated: bool) -> tuple[str, list[str], bool, bool]:
    ev = copy.deepcopy(case["evidence"])
    det = deterministic_assessment(copy.deepcopy(ev))
    trig = selective_trigger(ev, det)["invoke"]
    calls, recovered = [], False
    if trig or not gated:
        for f in missing_fields(ev):
            tool = FIELD_TO_TOOL.get(f)
            if tool and tool in (case.get("available_tools") or []) and tool not in calls:
                calls.append(tool)
                if tool == case.get("expected_tool"):
                    ev = merge_tool_result(ev, copy.deepcopy(case["tool_result"]))
                    recovered = True
    return str(deterministic_assessment(ev)["selected_action"]), calls, trig, recovered


def main() -> None:
    OUT.mkdir(exist_ok=True)
    rows = []
    for study, f in (("primary", "cases_v1.yaml"), ("replication", "cases_replication_v1.yaml")):
        for c in yaml.safe_load((ROOT / "agentic_experiments" / f).read_text())["cases"]:
            b1 = str(deterministic_assessment(copy.deepcopy(c["evidence"]))["selected_action"])
            a2a, calls_a, trig, rec_a = fetch_and_decide(c, gated=False)
            a2b, calls_b, _, rec_b = fetch_and_decide(c, gated=True)
            adm = c["admissible_actions"]
            rows.append({"study": study, "case": c["id"], "stratum": c["stratum"], "admissible": "|".join(adm),
                         "admissible_set_size": len(adm), "B1_action": b1, "B1_ok": b1 in adm,
                         "B2a_action": a2a, "B2a_ok": a2a in adm, "B2a_tool_calls": len(calls_a), "B2a_recovered_evidence": rec_a,
                         "B2b_action": a2b, "B2b_ok": a2b in adm, "B2b_tool_calls": len(calls_b), "B2b_recovered_evidence": rec_b,
                         "hybrid_trigger_fires": trig, "expected_tool": c.get("expected_tool") or ""})
    with (OUT / "autofetch_cases.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    summ = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    for r in rows:
        for key in (r["study"], "combined"):
            for grp in (r["stratum"], "ALL"):
                s = summ[key][grp]
                s["n"] += 1
                for arm in ("B1", "B2a", "B2b"):
                    s[f"{arm}_ok"] += r[f"{arm}_ok"]
                s["B2a_tool_calls"] += r["B2a_tool_calls"]; s["B2b_tool_calls"] += r["B2b_tool_calls"]
                s["B2a_recovered"] += r["B2a_recovered_evidence"]; s["B2b_recovered"] += r["B2b_recovered_evidence"]
    (OUT / "autofetch_summary.json").write_text(json.dumps(summ, indent=2))
    for key in ("primary", "replication", "combined"):
        print(key, {g: dict(v) for g, v in summ[key].items()})


if __name__ == "__main__":
    main()
