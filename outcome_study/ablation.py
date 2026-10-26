"""Component ablation of AAF's rules on the measured-outcome studies (no model calls). Each variant re-decides every
committed full-evidence snapshot with one part of rule set v1.4 removed:
  full          v1.4 (= B1-14)
  -R1'          no release-regression rule (v1.0 arbitration + R2)
  -R2           no security-over-cost rule
  -R1'-R2       v1.0 rules (= B1)
  -baseline     R1 without the pre-release baseline (v1.2's R1, = B1-12)
  -interaction  utility/eligible-set action without cross-domain interaction arbitration (+R1', R2)
  -interaction-R1'-R2  utility action alone
  single-domain the best of the four siloed v1.4 experts on each study (oracle-free upper bound on one domain)
Reports O1 / O2 accuracy on fault cases, overall and on discriminating cases, with exact McNemar of full vs each variant."""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agentic_experiments import rules_v14 as R  # noqa: E402
from agentic_experiments.rules_v12 import deterministic_assessment_v12  # noqa: E402
from evidence.schema import bool_evidence  # noqa: E402
from orchestrator.severity import severity_scores  # noqa: E402
from outcome_study.analyze_gpt import load, tests  # noqa: E402

FROZEN = {"p95_floor": 450, "factor": 2, "err_floor": 8, "err_rise": 5, "restarts": 3}


def v14(ev: dict, r1=True, r2=True, interaction=True) -> str:
    det = R._BASE_ASSESSMENT(ev)
    sev = severity_scores(ev)
    action = det["selected_action"] if interaction else det["base_action"]
    deploy, sre = ev.get("deploy", {}) or {}, ev.get("sre", {}) or {}
    rr = bool_evidence(deploy, "recent_release")
    restarts = R._num(deploy, "restart_loops") or 0.0
    p95, err = R._num(sre, "p95_latency_ms"), R._num(sre, "error_rate_pct")
    p95_0, err_0 = R._num(sre, "p95_pre_release_ms"), R._num(sre, "error_rate_pre_release_pct")
    if p95_0 is None and err_0 is None:
        regression = sev["reliability"] > 0
    else:
        regression = bool((p95 is not None and p95_0 is not None and p95 >= 450 and p95 >= 2 * p95_0)
                          or (err is not None and err_0 is not None and err >= 8 and err >= err_0 + 5))
    if r1 and rr.usable and rr.value and (regression or restarts >= 3):
        return "Rollback to stable deployment"
    if r2 and sev["security"] > 0 and action in R.COST_ONLY:
        return "Patch or block release"
    return str(action)


VARIANTS = {
    "full": lambda ev: v14(ev),
    "-R1'": lambda ev: v14(ev, r1=False),
    "-R2": lambda ev: v14(ev, r2=False),
    "-R1'-R2": lambda ev: v14(ev, r1=False, r2=False),
    "-baseline": lambda ev: str(deterministic_assessment_v12(copy.deepcopy(ev))["selected_action"]),
    "-interaction": lambda ev: v14(ev, interaction=False),
    "-interaction-R1'-R2": lambda ev: v14(ev, r1=False, r2=False, interaction=False),
}


def main() -> None:
    res = {}
    for st in ("outcome_study14", "outcome_study15", "outcome_study16"):
        if not (ROOT / st / "decisions" / "variants.json").exists():
            continue
        A, table, dec, ok, maj, _ = load(st)
        variants = {v["id"]: v for v in json.loads((ROOT / st / "decisions" / "variants.json").read_text())}
        faults = [c for c in sorted(table) if c.split("-")[1][0] in "RPSKDX" and c in variants]
        disc = [c for c in faults if len(table[c]["O1"]) < 5]
        acts = {n: {c: f(copy.deepcopy(variants[c]["evidence"])) for c in faults} for n, f in VARIANTS.items()}
        okv = lambda n, c, o="O1": A.STATE_OF.get(acts[n][c]) in table[c][o]
        siloed = [d for d in ("D-DevOps-14", "D-SRE-14", "D-FinOps-14", "D-Sec-14") if d in dec] or ["D-DevOps", "D-SRE", "D-FinOps", "D-Sec"]
        best = max(siloed, key=lambda d: sum(ok(d, c) for c in faults))
        acts["single-domain(best)"] = {c: maj(dec[best][c]) for c in faults}
        r = {"n_faults": len(faults), "n_discriminating": len(disc), "best_single_domain": best,
             "O1": {n: sum(okv(n, c) for c in faults) for n in acts}, "O2": {n: sum(okv(n, c, "O2") for c in faults) for n in acts},
             "O1_discriminating": {n: sum(okv(n, c) for c in disc) for n in acts},
             "check_full_equals_B1-14": all(acts["full"][c] == maj(dec["B1-14"][c]) for c in faults),
             "tests_full_vs": tests(lambda n, c: okv(n, c), "full", [n for n in acts if n != "full"], faults)}
        res[st] = r
        print(st, r["O1"], r["check_full_equals_B1-14"])
    (ROOT / "outcome_study" / "ablation.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
