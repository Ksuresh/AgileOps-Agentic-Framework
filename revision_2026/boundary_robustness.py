"""R6-7 / R1-2: parameter robustness on BOUNDARY cases (deterministic, no LLM).

Method fixed before running:
1. Reproduce the manuscript's one-at-a-time sweep on the 30 controlled scenarios (seed 42) and report, at the default
   parameters, how close each scenario is to a decision boundary: the utility margin between the best and second-best
   eligible action, and the distance of the consensus score from tau = 0.65.
2. Mine boundary cases from the same generator and noise model with fresh seeds 1..300 (the 30 scenario designs with
   jitter): keep a scenario if (a) its default utility margin is < 0.05 (weight-sensitive) or (b) its consensus score is
   within 0.03 of tau (tau-sensitive). Select up to 6 of each kind, seeded (20261022), at most one per design.
3. Re-run the same parameter grids on the boundary set and report action stability vs. default, and which parameter
   values flip which cases. Oracle agreement is reported too.
"""
from __future__ import annotations

import csv
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.parameter_sensitivity_2026 import DEFAULT, GRIDS, configured, run_rows, selected_action  # noqa: E402
from scenario_generator.generate import generate_scenarios  # noqa: E402

OUT = Path(__file__).resolve().parent / "results"
NOISE = {"missing_evidence_prob": 0.20, "contradiction_prob": 0.10, "metric_jitter_pct": 0.05}


def margin(row: dict) -> float:
    c = sorted((float(x["utility"]) for x in row.get("utility", {}).get("all_candidates") or row.get("utility", {}).get("candidates") or []), reverse=True)
    return round(c[0] - c[1], 4) if len(c) > 1 else 9.0


def oracle_ok(row: dict) -> bool:
    gt = row.get("ground_truth") or {}
    adm = gt.get("admissible_actions") or [gt.get("expected_action") or gt.get("recommended_action")]
    return selected_action(row) in adm


def sweep(base: list[dict]) -> tuple[list[dict], list[dict]]:
    d_rows = run_rows(configured(base, "lambda", DEFAULT["lambda"]))
    d_act = [selected_action(r) for r in d_rows]
    out, flips = [], []
    for fam, vals in GRIDS.items():
        for v in vals:
            rows = run_rows(configured(base, fam, v))
            acts = [selected_action(r) for r in rows]
            label = "/".join(f"{x:.2f}" for x in v) if fam == "utility_weights" else f"{float(v):.2f}"
            out.append({"family": fam, "value": label, "n": len(rows), "stability_vs_default": round(sum(a == b for a, b in zip(d_act, acts)) / len(acts), 3),
                        "oracle_agreement": round(sum(oracle_ok(r) for r in rows) / len(rows), 3)})
            for sc, a, b in zip(base, d_act, acts):
                if a != b:
                    flips.append({"case": sc["scenario_id"], "family": fam, "value": label, "default_action": a, "new_action": b})
    return out, flips


def main() -> None:
    OUT.mkdir(exist_ok=True)
    base = generate_scenarios(seed=42, noise=NOISE)
    rows = run_rows(configured(base, "lambda", DEFAULT["lambda"]))
    orig_prox = [{"case": r["scenario_id"], "utility_margin": margin(r), "consensus": r["consensus_score"],
                  "consensus_minus_tau": round(float(r["consensus_score"]) - DEFAULT["tau"], 3)} for r in rows]
    orig_sweep, orig_flips = sweep(base)

    cands_u, cands_t = [], []
    for seed in range(1, 301):
        scs = generate_scenarios(seed=seed, noise=NOISE)
        rr = run_rows(configured(scs, "lambda", DEFAULT["lambda"]))
        for sc, r in zip(scs, rr):
            sc = dict(sc, scenario_id=f"{sc['scenario_id']}@s{seed}")
            if margin(r) < 0.05:
                cands_u.append((sc, margin(r)))
            if abs(float(r["consensus_score"]) - DEFAULT["tau"]) <= 0.03 + 1e-9:
                cands_t.append((sc, float(r["consensus_score"])))
    rng = random.Random(20261022)

    def pick(cands, k):
        rng.shuffle(cands)
        seen, out = set(), []
        for sc, _ in cands:
            design = sc["scenario_id"].split("@")[0]
            if design not in seen:
                seen.add(design); out.append(sc)
            if len(out) == k:
                break
        return out
    bu, bt = pick(cands_u, 6), pick(cands_t, 6)
    boundary = bu + [s for s in bt if s["scenario_id"] not in {x["scenario_id"] for x in bu}]
    b_sweep, b_flips = sweep(boundary)
    b_rows = run_rows(configured(boundary, "lambda", DEFAULT["lambda"]))
    b_prox = [{"case": r["scenario_id"], "kind": "utility-near-tie" if s in bu else "consensus-near-tau", "utility_margin": margin(r),
               "consensus": r["consensus_score"], "default_action": selected_action(r), "oracle_ok": oracle_ok(r)} for s, r in zip(boundary, b_rows)]
    res = {"design": __doc__, "default_parameters": DEFAULT,
           "original_30": {"min_utility_margin": min(p["utility_margin"] for p in orig_prox),
                           "n_margin_below_0.05": sum(p["utility_margin"] < 0.05 for p in orig_prox),
                           "n_consensus_within_0.03_of_tau": sum(abs(p["consensus_minus_tau"]) <= 0.03 + 1e-9 for p in orig_prox),
                           "sweep": orig_sweep, "flips": orig_flips},
           "boundary_mining": {"seeds": "1..300", "utility_near_tie_candidates": len(cands_u), "consensus_near_tau_candidates": len(cands_t),
                               "selected": len(boundary)},
           "boundary_set": {"cases": b_prox, "sweep": b_sweep, "flips": b_flips,
                            "cases_that_flip_under_any_tested_value": sorted({f["case"] for f in b_flips})}}
    (OUT / "boundary_robustness.json").write_text(json.dumps(res, indent=1, default=str))
    with (OUT / "boundary_sweep.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["set", "family", "value", "n", "stability_vs_default", "oracle_agreement"])
        w.writeheader()
        for s, rows_ in (("original_30", orig_sweep), ("boundary", b_sweep)):
            for r in rows_:
                w.writerow({"set": s, **r})
    print(json.dumps({k: v for k, v in res["original_30"].items() if k not in ("sweep", "flips")}, indent=1))
    print("orig flips", len(orig_flips))
    print(json.dumps(res["boundary_mining"]), json.dumps(b_prox, indent=0))
    for r in b_sweep:
        print(r)
    print("flipping cases:", res["boundary_set"]["cases_that_flip_under_any_tested_value"])


if __name__ == "__main__":
    main()
