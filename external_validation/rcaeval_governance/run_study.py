"""RCAEval RE2 governance study — see PROTOCOL.md (pre-registered). Deterministic, no LLM."""
from __future__ import annotations

import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from external_validation.rcaeval.adapter import adapt  # frozen
from orchestrator.cross_domain_v2 import apply_interaction_policy_v2
from orchestrator.utility import choose_action_details

REPO_ID = "phamquiluan/RCAEval"
WEIGHTS = (0.45, 0.20, 0.35)  # frozen RCAEval configuration
OUT = Path(__file__).resolve().parent / "results"
OBSERVE = "No action (observe)"
RESOURCE = {"cpu", "mem", "disk", "socket"}
NETWORK = {"delay", "loss"}
GATE = 1.5


def dl(path: str) -> Path:
    return Path(hf_hub_download(repo_id=REPO_ID, repo_type="dataset", filename=path))


def admissible(window: str, fault: str) -> set[str]:
    if window == "control":
        return {OBSERVE}
    if fault in RESOURCE:
        return {"Mitigate and monitor", "Scale adjustment"}
    if fault in NETWORK:
        return {"Mitigate and monitor"}
    raise ValueError(fault)


def aaf_decision(metrics: pd.DataFrame, t: int) -> tuple[str, dict, dict]:
    telemetry, diag = adapt(metrics, t, logs=None)
    base = choose_action_details(telemetry, WEIGHTS)
    final = apply_interaction_policy_v2(telemetry, str(base["selected_action"]))
    return str(final["selected_action"]), telemetry, diag


def baseline_decision(diag: dict) -> tuple[str, float | None]:
    ratios = []
    for key in ("latency90_largest_ratio", "cpu_largest_ratio"):
        c = diag.get(key)
        if c and c.get("ratio") is not None:
            ratios.append(float(c["ratio"]))
    mem = (diag.get("auxiliary_evidence") or {}).get("memory_largest_ratio")
    if mem and mem.get("ratio") is not None:
        ratios.append(float(mem["ratio"]))
    top = max(ratios) if ratios else None
    return ("Mitigate and monitor" if top is not None and top >= GATE else OBSERVE), top


def dev_case_ids(index: pd.DataFrame) -> set[str]:
    """Replicates run_heldout.pilot_case_ids on RE2-TT (first two sorted cases per fault)."""
    tt = index[index["dataset"].astype(str).str.upper() == "RE2-TT"]
    out: set[str] = set()
    for fault in sorted(tt["fault"].astype(str).unique()):
        out.update(tt[tt["fault"].astype(str) == fault].sort_values("case").head(2)["case"].astype(str))
    return out


def wilson(k: int, n: int) -> list[float] | None:
    if n == 0:
        return None
    z, p = 1.96, k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(100 * (c - h), 1), round(100 * (c + h), 1)]


def rate(rows: list[dict], key: str) -> dict:
    k, n = sum(1 for r in rows if r[key]), len(rows)
    return {"k": k, "n": n, "pct": round(100 * k / n, 1) if n else None, "wilson95": wilson(k, n)}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    idx = pd.read_parquet(dl("cases.parquet"))
    re2 = idx[idx["suite"].astype(str) == "RE2"].sort_values("case")
    dev = dev_case_ids(idx)
    rows: list[dict] = []
    for _, row in re2.iterrows():
        case, fault = str(row["case"]), str(row["fault"])
        inject = int(dl(f"{case}/inject_time.txt").read_text().strip())
        metrics = pd.read_parquet(dl(f"{case}/metrics.parquet"))
        for window, t in (("fault", inject), ("control", inject - 300)):
            aaf, telemetry, diag = aaf_decision(metrics, t)
            base, top_ratio = baseline_decision(diag)
            adm = admissible(window, fault)
            cpu = diag.get("cpu_largest_ratio") or {}
            rows.append({
                "case": case, "dataset": str(row["dataset"]), "system": str(row["system_name"]),
                "root_cause_service": str(row["root_cause_service"]), "fault": fault,
                "repetition": str(row["repetition"]), "pair": f"{row['dataset']}:{row['root_cause_service']}:{fault}",
                "window": window, "split": "tt_dev" if case in dev else "heldout",
                "aaf_action": aaf, "baseline_action": base, "admissible": "|".join(sorted(adm)),
                "aaf_ok": aaf in adm, "baseline_ok": base in adm,
                "aaf_active": aaf != OBSERVE, "baseline_active": base != OBSERVE,
                "aaf_saturation_input": telemetry.get("sre", {}).get("saturation_pct"),
                "aaf_p95_proxy_input": telemetry.get("sre", {}).get("p95_latency_ms"),
                "cpu_top_metric": cpu.get("column"), "cpu_top_post": cpu.get("post_median"), "cpu_top_ratio": cpu.get("ratio"),
                "baseline_top_ratio": top_ratio,
            })

    df = pd.DataFrame(rows)
    csv = OUT / "window_results.csv"
    df.to_csv(csv, index=False)
    held = [r for r in rows if r["split"] == "heldout"]

    def block(rs: list[dict]) -> dict:
        f = [r for r in rs if r["window"] == "fault"]
        c = [r for r in rs if r["window"] == "control"]
        out = {}
        for arm in ("aaf", "baseline"):
            out[arm] = {
                "admissible_all_windows": rate(rs, f"{arm}_ok"),
                "detection_on_fault_windows": rate(f, f"{arm}_active"),
                "admissible_on_fault_windows": rate(f, f"{arm}_ok"),
                "false_alarm_on_control_windows": rate(c, f"{arm}_active"),
                "actions_fault": dict(Counter(r[f"{arm}_action"] for r in f)),
                "actions_control": dict(Counter(r[f"{arm}_action"] for r in c)),
            }
        a = [r["aaf_ok"] for r in rs]; b = [r["baseline_ok"] for r in rs]
        n10 = sum(x and not y for x, y in zip(a, b)); n01 = sum(y and not x for x, y in zip(a, b))
        out["paired_aaf_vs_baseline"] = {"aaf_only": n10, "baseline_only": n01,
                                         "exact_p": round(float(binomtest(min(n10, n01), n10 + n01, 0.5).pvalue), 6) if n10 + n01 else 1.0}
        return out

    by_fault = {f: block([r for r in held if r["fault"] == f]) for f in sorted({r["fault"] for r in held})}
    by_system = {s: block([r for r in held if r["system"] == s]) for s in sorted({r["system"] for r in held})}
    pairs = defaultdict(list)
    for r in held:
        if r["window"] == "fault":
            pairs[r["pair"]].append(r)
    pair_majority = {arm: {"k": sum(1 for v in pairs.values() if sum(x[f"{arm}_ok"] for x in v) * 2 > len(v)), "n": len(pairs)}
                     for arm in ("aaf", "baseline")}
    cpu_fault = [r for r in held if r["window"] == "fault" and r["fault"] == "cpu"]
    summary = {
        "protocol": "external_validation/rcaeval_governance/PROTOCOL.md",
        "windows_total": len(rows), "heldout_windows": len(held), "tt_dev_windows": len(rows) - len(held),
        "primary_heldout": block(held),
        "tt_dev_cases_separately": block([r for r in rows if r["split"] == "tt_dev"]),
        "by_fault_heldout": by_fault, "by_system_heldout": by_system,
        "fault_pair_majority_heldout": pair_majority,
        "cpu_fault_diagnostic": {
            "n": len(cpu_fault),
            "aaf_saturation_input_values": sorted(round(float(r["aaf_saturation_input"]), 1) for r in cpu_fault if r["aaf_saturation_input"] is not None),
            "cpu_fault_windows_without_saturation_input": sum(1 for r in cpu_fault if r["aaf_saturation_input"] is None),
            "aaf_thresholds": {"saturation_elevated": 85, "saturation_severe": 90},
        },
        "csv_sha256": hashlib.sha256(csv.read_bytes()).hexdigest(),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary["primary_heldout"], indent=2, default=str))


if __name__ == "__main__":
    main()
