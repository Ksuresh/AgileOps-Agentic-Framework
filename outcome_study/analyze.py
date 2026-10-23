"""Study 9 analysis (PROTOCOL.md): outcome oracle O1/O2/O3 from measured outcomes (mean over replicates), decider
correctness, and the pre-specified exact McNemar tests with Holm adjustment."""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from external_validation.expert_study.analyze import holm, mcnemar, wilson  # noqa: E402

HERE = ROOT / "outcome_study"
STATE_OF = {"Rollback to stable deployment": "rollback", "Block release and fix pipeline": "rollback", "Patch or block release": "patch",
            "Mitigate and monitor": "mitigate", "Scale adjustment": "scale", "No action (observe)": "none",
            "Review scaling policy": "none", "Escalate for evidence/human review": "none"}
SILOED = ["D-DevOps", "D-SRE", "D-FinOps", "D-Sec", "L-DevOps", "L-SRE", "L-FinOps", "L-Sec"]


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def outcome_table(cal: dict) -> dict:
    p95_max = 2 * cal["healthy_p95_median_ms"]
    reps = defaultdict(list)
    for p in sorted((HERE / "measurements").glob("rep*/*.json")):
        d = json.loads(p.read_text())
        reps[d["case"]["id"]].append(d)
    table = {}
    for cid, ds in reps.items():
        acts = {}
        for a in ("none", "rollback", "patch", "mitigate", "scale"):
            ms = [d["outcomes"][a] for d in ds]
            p95 = mean([m["http"]["latency_p95_ms"] for m in ms])
            err = mean([m["http"]["error_rate_pct"] for m in ms])
            sec = any(m["critical_cves_new"] > 0 or m["secret_in_config"] for m in ms)
            foot = mean([m["footprint_cpu_m"] for m in ms])
            v = {"reliability": bool(p95 is None or p95 > p95_max or err > 1.0), "security": sec, "cost": foot > 300.0}
            acts[a] = {"violations": v, "n_violations": sum(v.values()), "delivered": all(m["feature_delivered"] for m in ms),
                       "footprint": foot, "p95": p95, "err": err}
        best = min(x["n_violations"] for x in acts.values())
        o1 = sorted(a for a, x in acts.items() if x["n_violations"] == best)
        key = lambda a: (not acts[a]["delivered"], acts[a]["footprint"])
        o2_best = min(key(a) for a in o1)
        o2 = sorted(a for a in o1 if key(a) == o2_best)
        vec = {a: (x["violations"]["reliability"], x["violations"]["security"], x["violations"]["cost"]) for a, x in acts.items()}
        o3 = sorted(a for a in acts if not any(all(vb <= va for vb, va in zip(vec[b], vec[a])) and vec[b] != vec[a] for b in acts))
        table[cid] = {"actions": acts, "O1": o1, "O2": o2, "O3": o3, "n_reps": len(ds), "min_violations": best}
    return table


def main() -> None:
    cal = json.loads((HERE / "calibration" / "calibration.json").read_text())
    cases = {c["id"]: c for c in yaml.safe_load((HERE / "cases.yaml").read_text())["cases"]}
    table = outcome_table(cal)
    dec = defaultdict(lambda: defaultdict(list))  # decider -> variant case id -> actions
    for line in (HERE / "decisions" / "deterministic.jsonl").read_text().splitlines():
        r = json.loads(line)
        dec[r["decider"]][r["case"]].append(r["action"])
    for f in sorted((HERE / "decisions").glob("llm_shard_*.jsonl")):
        for line in f.read_text().splitlines():
            r = json.loads(line)
            if r.get("error"):
                continue
            dec[r["decider"]][r["case"]].append(r["action"])
            if r["decider"] == "B5":
                dec["B3"][r["case"]].append(r["b3_action"])
                for o in r.get("agent_outputs") or []:
                    if o.get("proposed_action") and o.get("agent_type"):
                        dec["L-" + {"DevSecOps": "Sec"}.get(o["agent_type"], o["agent_type"])][r["case"]].append(o["proposed_action"])

    def maj(xs):
        c = Counter(xs).most_common()
        return sorted(a for a, n in c if n == c[0][1])[0] if c else None

    def correct(decider, vid, oracle="O1"):
        base = vid[:-2] if vid.endswith("-W") else vid
        a = maj(dec[decider].get(vid) or [])
        return a is not None and STATE_OF.get(a) in table[base][oracle]

    full = [c for c in table if c in cases]
    compound = [c for c in full if c.startswith("OX")]
    faults = [c for c in full if not c.startswith("OC")]
    withheld = [c + "-W" for c in full if (c + "-W") in dec["B1"]]
    deciders = sorted(dec)
    res = {"n_cases_measured": len(table), "reps": Counter(t["n_reps"] for t in table.values()),
           "reliability_threshold_p95_ms": 2 * cal["healthy_p95_median_ms"], "oracle": {c: {k: table[c][k] for k in ("O1", "O2", "O3", "min_violations")} for c in table}}
    for oracle in ("O1", "O2", "O3"):
        res[f"accuracy_{oracle}"] = {g: {d: sum(correct(d, c, oracle) for c in ids) for d in deciders} | {"n": len(ids)}
                                     for g, ids in (("compound", compound), ("faults", faults), ("controls", [c for c in full if c.startswith("OC")]), ("withheld", withheld))}

    def tests(a, others, ids, oracle="O1"):
        out = {b: mcnemar({c: correct(a, c, oracle) for c in ids}, {c: correct(b, c, oracle) for c in ids}, ids) for b in others if b in dec}
        adj = holm([v["p_exact"] for v in out.values()])
        for (b, v), p in zip(out.items(), adj):
            v["p_holm"] = p
        return out
    res["H9_1_compound_vs_siloed"] = tests("AAF", SILOED, compound)
    res["H9_2_faults_vs_LLM_and_policies"] = tests("AAF", ["B5", "B6", "Dominant", "AlwaysRollback"], faults)
    res["H9_3_withheld"] = tests("AAF", ["B1", "B2", "B3"], withheld)
    res["per_case"] = {c: {"O1": table[c.replace('-W', '')]["O1"], **{d: maj(dec[d].get(c) or []) for d in deciders}} for c in full + withheld}
    (HERE / "analysis.json").write_text(json.dumps(res, indent=2, default=str))
    print(json.dumps({k: res[k] for k in res if k.startswith(("accuracy_O1", "H9"))}, indent=2, default=str))


if __name__ == "__main__":
    main()
