"""Oracle-threshold sensitivity (post hoc, reported as such): re-score Studies 9, 12 (and 14 when present) with the
reliability factor (x healthy p95) in {1.5, 2, 3}, the error threshold in {0.5, 1, 2}% and the cost factor (x healthy
footprint) in {2, 3, 4}; report each decider's O1 correct count on fault cases and AAF vs. siloed-rule comparisons."""
from __future__ import annotations

import itertools
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from external_validation.expert_study.analyze import mcnemar  # noqa: E402

STATE_OF = {"Rollback to stable deployment": "rollback", "Block release and fix pipeline": "rollback", "Patch or block release": "patch",
            "Mitigate and monitor": "mitigate", "Scale adjustment": "scale", "No action (observe)": "none",
            "Review scaling policy": "none", "Escalate for evidence/human review": "none"}


def o1_sets(study: Path, rel_f: float, err_t: float, cost_f: float) -> dict:
    cal = json.loads((study / "calibration" / "calibration.json").read_text())
    p95_max = rel_f * cal["healthy_p95_median_ms"]
    reps = defaultdict(list)
    for p in sorted((study / "measurements").glob("rep*/*.json")):
        d = json.loads(p.read_text()); reps[d["case"]["id"]].append(d)
    out = {}
    for cid, ds in reps.items():
        nv = {}
        for a in ("none", "rollback", "patch", "mitigate", "scale"):
            ms = [d["outcomes"][a] for d in ds]
            p95 = sum(m["http"]["latency_p95_ms"] for m in ms) / len(ms)
            err = sum(m["http"]["error_rate_pct"] for m in ms) / len(ms)
            foot = sum(m["footprint_cpu_m"] for m in ms) / len(ms)
            sec = any(m["critical_cves_new"] > 0 or m["secret_in_config"] for m in ms)
            nv[a] = int(p95 > p95_max or err > err_t) + int(sec) + int(foot > cost_f * 100.0)
        best = min(nv.values())
        out[cid] = {a for a, n in nv.items() if n == best}
    return out


def decisions(study: Path) -> dict:
    dec = defaultdict(lambda: defaultdict(list))
    for line in (study / "decisions" / "deterministic.jsonl").read_text().splitlines():
        r = json.loads(line); dec[r["decider"]][r["case"]].append(r["action"])
    for f in sorted((study / "decisions").glob("llm_shard_*.jsonl")):
        for line in f.read_text().splitlines():
            r = json.loads(line)
            if r.get("error") is None:
                dec[r["decider"]][r["case"]].append(r["action"])
    return {d: {c: Counter(v).most_common(1)[0][0] for c, v in cs.items()} for d, cs in dec.items()}


def main() -> None:
    studies = [s for s in ("outcome_study", "outcome_study12", "outcome_study14") if (ROOT / s / "decisions" / "deterministic.jsonl").exists()]
    grid = list(itertools.product((1.5, 2.0, 3.0), (0.5, 1.0, 2.0), (2.0, 3.0, 4.0)))
    res = {}
    for s in studies:
        sd = ROOT / s
        dec = decisions(sd)
        aaf = "AAF-14" if "AAF-14" in dec else ("AAF-12" if "AAF-12" in dec else "AAF")
        rows = []
        for rf, et, cf in grid:
            o1 = o1_sets(sd, rf, et, cf)
            faults = [c for c in o1 if not (c.startswith("OC") or c.startswith("PC") or "-C" in c or "-L" in c)]
            ok = {d: {c: STATE_OF.get(dec[d].get(c)) in o1[c] for c in faults} for d in dec}
            row = {"rel_factor": rf, "err_pct": et, "cost_factor": cf, "n": len(faults),
                   "correct": {d: sum(v.values()) for d, v in ok.items()}}
            for base in ("AAF", aaf):
                for silo in ("D-DevOps", "D-SRE", "D-FinOps", "D-Sec"):
                    t = mcnemar(ok[base], ok[silo], faults)
                    row[f"{base}_vs_{silo}"] = [t["a_only"], t["b_only"], round(t["p_exact"], 4)]
            rows.append(row)
        res[s] = rows
        best_silo = [max(r["correct"][x] for x in ("D-DevOps", "D-SRE", "D-FinOps", "D-Sec")) for r in rows]
        print(s, "configs:", len(rows), "| AAF correct range", min(r["correct"]["AAF"] for r in rows), "-", max(r["correct"]["AAF"] for r in rows),
              "| best silo range", min(best_silo), "-", max(best_silo),
              "| configs where AAF >= every silo:", sum(r["correct"]["AAF"] >= b for r, b in zip(rows, best_silo)),
              "| primary AAF variant", aaf, sum(r["correct"][aaf] >= b for r, b in zip(rows, best_silo)))
    (ROOT / "outcome_study" / "sensitivity.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
