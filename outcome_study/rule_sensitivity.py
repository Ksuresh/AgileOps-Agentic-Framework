"""Sensitivity of AAF's outcome accuracy to its own release-regression thresholds (no model calls).
Rule set v1.4's R1' uses four hand-set numbers (p95 floor 450 ms, factor 2 x pre-release p95, error floor 8%, error rise
5 points) and a restart count (3). This script re-decides every committed full-evidence snapshot of the outcome studies
with each combination of p95 floor {350, 450, 550}, factor {1.5, 2, 3}, error floor {5, 8, 12}, error rise {3, 5, 8} and
restarts {2, 3, 5} (243 settings; the frozen setting is one of them) and reports, per study, the O1 accuracy of the rules
alone (B1-14) on fault cases and how often it stays >= each comparison decider (siloed experts, B5, B6, AlwaysRollback).
Usage: python outcome_study/rule_sensitivity.py [study ...] (default: every outcome study 14-16 with decisions)."""
from __future__ import annotations

import copy
import importlib
import itertools
import json
import os
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agentic_experiments import rules_v14 as R  # noqa: E402
from evidence.schema import bool_evidence  # noqa: E402
from orchestrator.severity import severity_scores  # noqa: E402

GRID = {"p95_floor": [350, 450, 550], "factor": [1.5, 2, 3], "err_floor": [5, 8, 12], "err_rise": [3, 5, 8], "restarts": [2, 3, 5]}
FROZEN = {"p95_floor": 450, "factor": 2, "err_floor": 8, "err_rise": 5, "restarts": 3}
FAULT_GROUPS = ("R", "P", "S", "K", "D", "X")
COMPARE = ["D-DevOps", "D-SRE", "D-FinOps", "D-Sec", "D-DevOps-14", "D-SRE-14", "D-FinOps-14", "D-Sec-14", "B5", "B6", "AlwaysRollback"]


def assess(ev: dict, p: dict) -> str:
    det = R._BASE_ASSESSMENT(ev)
    sev = severity_scores(ev)
    deploy, sre = ev.get("deploy", {}) or {}, ev.get("sre", {}) or {}
    rr = bool_evidence(deploy, "recent_release")
    restarts = R._num(deploy, "restart_loops") or 0.0
    p95, err = R._num(sre, "p95_latency_ms"), R._num(sre, "error_rate_pct")
    p95_0, err_0 = R._num(sre, "p95_pre_release_ms"), R._num(sre, "error_rate_pre_release_pct")
    if p95_0 is None and err_0 is None:
        regression = sev["reliability"] > 0
    else:
        regression = bool((p95 is not None and p95_0 is not None and p95 >= p["p95_floor"] and p95 >= p["factor"] * p95_0)
                          or (err is not None and err_0 is not None and err >= p["err_floor"] and err >= err_0 + p["err_rise"]))
    action = det["selected_action"]
    if rr.usable and rr.value and (regression or restarts >= p["restarts"]):
        return "Rollback to stable deployment"
    if sev["security"] > 0 and action in R.COST_ONLY:
        return "Patch or block release"
    return str(action)


def study(st: str) -> dict:
    os.environ["OUTCOME_DIR"] = st
    from outcome_study import analyze as A
    importlib.reload(A)
    here = ROOT / st
    table = A.outcome_table(json.loads((here / "calibration" / "calibration.json").read_text()))
    variants = {v["id"]: v for v in json.loads((here / "decisions" / "variants.json").read_text())}
    faults = [c for c in sorted(table) if c.split("-")[1][0] in FAULT_GROUPS and c in variants]
    dec = defaultdict(lambda: defaultdict(list))
    for line in (here / "decisions" / "deterministic.jsonl").read_text().splitlines():
        r = json.loads(line); dec[r["decider"]][r["case"]].append(r["action"])
    for f in sorted((here / "decisions").glob("llm_shard_*.jsonl")):
        for line in f.read_text().splitlines():
            r = json.loads(line)
            if not r.get("error"):
                dec[r["decider"]][r["case"]].append(r["action"])
    maj = lambda xs: sorted(a for a, n in Counter(xs).most_common() if n == Counter(xs).most_common()[0][1])[0] if xs else None
    ok = lambda a, c: A.STATE_OF.get(a) in table[c]["O1"]
    comp = {d: sum(ok(maj(dec[d][c]), c) for c in faults) for d in COMPARE if dec[d]}
    acc = {}
    for vals in itertools.product(*GRID.values()):
        p = dict(zip(GRID, vals))
        acc[vals] = sum(ok(assess(copy.deepcopy(variants[c]["evidence"]), p), c) for c in faults)
    xs = list(acc.values())
    frozen = acc[tuple(FROZEN[k] for k in GRID)]
    return {"n_faults": len(faults), "settings": len(xs), "frozen_setting_correct": frozen,
            "correct_min": min(xs), "correct_median": statistics.median(xs), "correct_max": max(xs),
            "share_settings_at_least_frozen_minus_2": round(sum(x >= frozen - 2 for x in xs) / len(xs), 3),
            "comparators_correct": comp,
            "share_settings_ge_comparator": {d: round(sum(x >= v for x in xs) / len(xs), 3) for d, v in comp.items()},
            "share_settings_gt_comparator": {d: round(sum(x > v for x in xs) / len(xs), 3) for d, v in comp.items()},
            "by_parameter": {k: {str(v): statistics.mean(a for vals, a in acc.items() if vals[i] == v) for v in GRID[k]}
                             for i, k in enumerate(GRID)}}


def main() -> None:
    sts = sys.argv[1:] or [s for s in ("outcome_study14", "outcome_study15", "outcome_study16")
                           if (ROOT / s / "decisions" / "variants.json").exists() and (ROOT / s / "decisions" / "deterministic.jsonl").exists()]
    res = {"grid": GRID, "frozen": FROZEN, "studies": {s: study(s) for s in sts}}
    (ROOT / "outcome_study" / "rule_sensitivity.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
