"""AAF v1 vs v3 vs change baseline on RCAEval (PLAN.md H1/H2; RE2 = development check).

Usage: python external_validation/aaf_v3/run_rcaeval_v3.py RE2        (development: reproduces v1, shows v3 dev numbers)
       python external_validation/aaf_v3/run_rcaeval_v3.py RE1 RE3    (holdout)
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from external_validation.aaf_v3.rcaeval_v3 import decisions  # noqa: E402
from external_validation.rcaeval_governance.run_study import baseline_decision, rate  # noqa: E402
from external_validation.postmortems.run_study import holm, mcnemar  # noqa: E402

REPO = "phamquiluan/RCAEval"
OBSERVE = "No action (observe)"
RESOURCE, NETWORK, CODE = {"cpu", "mem", "disk", "socket"}, {"delay", "loss"}, {"f1", "f2", "f3", "f4", "f5"}
CODE_ADM = {"Rollback to stable deployment", "Block release and fix pipeline", "Patch or block release", "Mitigate and monitor"}
HERE = Path(__file__).resolve().parent


def dl(p: str) -> Path:
    return Path(hf_hub_download(REPO, p, repo_type="dataset"))


def admissible(window: str, fault: str) -> set[str] | None:
    if window == "control":
        return {OBSERVE}
    if fault in RESOURCE:
        return {"Mitigate and monitor", "Scale adjustment"}
    if fault in NETWORK:
        return {"Mitigate and monitor"}
    if fault in CODE:
        return CODE_ADM
    return None  # reported separately


def normalise_columns(m: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    """PLAN amendment A1: RE1-OB exposes service latency only as '<svc>_latency' (no '_latency-90'). In that case
    those columns are used as the latency family for every arm (format only; decided from column names)."""
    if any(c.endswith("_latency-90") for c in m.columns):
        return m, False
    ren = {c: c + "-90" for c in m.columns if c.endswith("_latency")}
    return (m.rename(columns=ren), True) if ren else (m, False)


def main() -> None:
    suites = sys.argv[1:] or ["RE2"]
    idx = pd.read_parquet(dl("cases.parquet"))
    rows = []
    for suite in suites:
        sub = idx[idx["suite"].astype(str) == suite].sort_values("case")
        for _, row in sub.iterrows():
            case, fault = str(row["case"]), str(row["fault"])
            inject = int(dl(f"{case}/inject_time.txt").read_text().strip())
            m, renamed = normalise_columns(pd.read_parquet(dl(f"{case}/metrics.parquet")))
            for window, t in (("fault", inject), ("control", inject - 300)):
                d = decisions(m, t)
                base, top = baseline_decision(d["diag"])
                adm = admissible(window, fault)
                rec = {"suite": suite, "case": case, "dataset": str(row["dataset"]), "system": str(row["system_name"]), "fault": fault,
                       "window": window, "latency_columns_renamed": renamed, "v1_action": d["v1"], "v3_action": d["v3"], "baseline_action": base,
                       "v3_relative_fired": json.dumps(d["v3_relative_fired"]), "admissible": "|".join(sorted(adm)) if adm else ""}
                for arm, act in (("v1", d["v1"]), ("v3", d["v3"]), ("baseline", base)):
                    rec[f"{arm}_ok"] = (act in adm) if adm else None
                    rec[f"{arm}_active"] = act != OBSERVE
                rows.append(rec)
    df = pd.DataFrame(rows)
    out = HERE / "results" / "rcaeval" / ("_".join(suites) + "_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "window_results.csv", index=False)

    def block(rs: list[dict]) -> dict:
        rs = [r for r in rs if r["v1_ok"] is not None]
        f = [r for r in rs if r["window"] == "fault"]; c = [r for r in rs if r["window"] == "control"]
        res = {arm: {"admissible_all": rate(rs, f"{arm}_ok"), "detection_fault": rate(f, f"{arm}_active"),
                     "admissible_fault": rate(f, f"{arm}_ok"), "false_alarm_control": rate(c, f"{arm}_active"),
                     "actions_fault": dict(Counter(r[f"{arm}_action"] for r in f))} for arm in ("v1", "v3", "baseline")}
        res["paired"] = {"v3_vs_v1": mcnemar([r["v3_ok"] for r in rs], [r["v1_ok"] for r in rs]),
                         "v3_vs_baseline": mcnemar([r["v3_ok"] for r in rs], [r["baseline_ok"] for r in rs])}
        return res

    recs = df.to_dict("records")
    summary = {"plan": "external_validation/aaf_v3/PLAN.md", "suites": suites, "windows": len(recs),
               "all": block(recs),
               "by_suite": {s: block([r for r in recs if r["suite"] == s]) for s in suites},
               "by_dataset": {s: block([r for r in recs if r["dataset"] == s]) for s in sorted(df["dataset"].unique())},
               "by_fault": {s: block([r for r in recs if r["fault"] == s]) for s in sorted(df["fault"].unique())},
               "unmapped_fault_windows": int(sum(1 for r in recs if r["v1_ok"] is None)),
               "v3_sha256": hashlib.sha256((HERE / "rcaeval_v3.py").read_bytes()).hexdigest(),
               "github_run_id": os.environ.get("GITHUB_RUN_ID")}
    summary["primary_holm"] = holm({f"{s}_v3_vs_v1": summary["by_suite"][s]["paired"]["v3_vs_v1"]["exact_p"] for s in suites})
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary["by_suite"], indent=1, default=str))


if __name__ == "__main__":
    main()
