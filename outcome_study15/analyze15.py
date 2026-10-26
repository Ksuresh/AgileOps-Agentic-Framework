"""Study 15 analysis (PROTOCOL.md H15.1-H15.7 and the descriptive analyses). Written before any Study 15 data existed."""
from __future__ import annotations

import importlib
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from external_validation.expert_study.analyze import holm, mcnemar, wilson  # noqa: E402

SILOED = ["D-DevOps-14", "D-SRE-14", "D-FinOps-14", "D-Sec-14", "L-DevOps", "L-SRE", "L-FinOps", "L-Sec"]
SILOED_V10 = ["D-DevOps", "D-SRE", "D-FinOps", "D-Sec"]
FAULT_GROUPS = ("R", "P", "S", "K", "D", "X")
SERVICES = (("frontend", "UF"), ("productcatalogservice", "UP"), ("currencyservice", "UC"))
LLM = ("AAF-14", "B5", "B6")


def load(study: str):
    os.environ["OUTCOME_DIR"] = study
    from outcome_study import analyze as A
    importlib.reload(A)
    here = ROOT / study
    table = A.outcome_table(json.loads((here / "calibration" / "calibration.json").read_text()))
    dec = defaultdict(lambda: defaultdict(list))
    usage = defaultdict(lambda: defaultdict(list))
    for line in (here / "decisions" / "deterministic.jsonl").read_text().splitlines():
        r = json.loads(line); dec[r["decider"]][r["case"]].append(r["action"])
    for f in sorted((here / "decisions").glob("llm_shard_*.jsonl")):
        for line in f.read_text().splitlines():
            r = json.loads(line)
            if r.get("error"):
                continue
            dec[r["decider"]][r["case"]].append(r["action"])
            usage[r["decider"]]["llm_calls"].append(r.get("n_calls") or 0)
            usage[r["decider"]]["total_tokens"].append((r.get("usage") or {}).get("total_tokens") or 0)
            usage[r["decider"]]["invoked"].append(1 if r.get("agentic_invoked") else 0)
            if r["decider"] == "B5":
                dec["B3"][r["case"]].append(r["b3_action"])
                if r.get("b3_14_action"):
                    dec["B3-14"][r["case"]].append(r["b3_14_action"])
                for o in r.get("agent_outputs") or []:
                    if o.get("proposed_action") and o.get("agent_type"):
                        dec["L-" + {"DevSecOps": "Sec"}.get(o["agent_type"], o["agent_type"])][r["case"]].append(o["proposed_action"])

    def maj(xs):
        c = Counter(xs).most_common()
        return sorted(a for a, n in c if n == c[0][1])[0] if c else None

    def ok(d, vid, oracle="O1"):
        a = maj(dec[d].get(vid) or [])
        return a is not None and A.STATE_OF.get(a) in table[vid.replace("-W", "")][oracle]
    return A, table, dec, ok, maj, usage


def tests(ok, a, others, ids, oracle="O1", adjust=True):
    out = {b: mcnemar({c: ok(a, c, oracle) for c in ids}, {c: ok(b, c, oracle) for c in ids}, ids) for b in others}
    if adjust and len(out) > 1:
        for (b, v), p in zip(out.items(), holm([v["p_exact"] for v in out.values()])):
            v["p_holm"] = p
    return out


def ci(k, n):
    return {"k": k, "n": n, "pct": round(100 * k / n, 1) if n else None, "wilson95_pct": wilson(k, n)}


def main() -> None:
    A, table, dec, ok, maj, usage = load("outcome_study15")
    cases = sorted(table)
    group = lambda c: c.split("-")[1][0]
    faults = [c for c in cases if group(c) in FAULT_GROUPS]
    lows = [c for c in cases if group(c) == "L"]
    withheld = [c + "-W" for c in cases if (c + "-W") in dec["B1"]]
    deciders = sorted(dec)
    res = {"n_cases": len(cases), "n_faults": len(faults), "n_low_risk": len(lows), "n_withheld": len(withheld),
           "runs": {d: dict(Counter(len(v) for v in dec[d].values())) for d in LLM}}
    for oracle in ("O1", "O2", "O3"):
        res[f"accuracy_{oracle}"] = {}
        for g, ids in (("faults", faults), *[(f"{s}_faults", [c for c in faults if c.startswith(p)]) for s, p in SERVICES],
                       *[(f"group_{x}", [c for c in cases if group(c) == x]) for x in ("C", "L", "R", "P", "S", "K", "D", "X")],
                       ("withheld", withheld)):
            res[f"accuracy_{oracle}"][g] = {d: sum(ok(d, c, oracle) for c in ids) for d in deciders} | {"n": len(ids)}
    res["H15_1_vs_siloed_v14"] = tests(ok, "AAF-14", SILOED, faults)
    res["H15_1b_vs_siloed_v10"] = tests(ok, "AAF-14", SILOED_V10, faults)
    res["H15_2_vs_LLM_and_policies"] = tests(ok, "AAF-14", ["B5", "B6", "AlwaysRollback"], faults)

    # H15.3 pooled with Study 14 (fault cases of both studies)
    _, t14, d14, ok14, _, _ = load("outcome_study14")
    _, table, dec, ok, maj, usage = load("outcome_study15")  # restore Study 15 module state
    pooled = defaultdict(dict)
    for c in sorted(t14):
        if c.split("-")[1][0] in FAULT_GROUPS:
            for d in ("AAF-14", "B5", "B6", "AlwaysRollback", "D-Sec"):
                pooled[d]["S14:" + c] = ok14(d, c)
    for c in faults:
        for d in ("AAF-14", "B5", "B6", "AlwaysRollback", "D-Sec"):
            pooled[d]["S15:" + c] = ok(d, c)
    ids = sorted(pooled["AAF-14"])
    res["H15_3_pooled_14_15"] = {"n": len(ids), "correct": {d: sum(v.values()) for d, v in pooled.items()},
                                 "tests": tests(lambda d, c, o="O1": pooled[d][c], "AAF-14", ["B6", "B5", "AlwaysRollback", "D-Sec"], ids)}

    # H15.4 risk-based release gate on L cases (O2), plus pooled with Study 14's L cases
    gate = ["AAF-14", "B5", "B6", "AlwaysRollback", "L-Sec", "D-Sec-14", "BlockAnyCVE", "B1-14"]
    res["H15_4_release_gate_O2"] = {"study15": {d: ci(sum(ok(d, c, "O2") for c in lows), len(lows)) for d in gate},
                                    "tests": tests(ok, "AAF-14", ["B5", "B6", "AlwaysRollback", "L-Sec"], lows, oracle="O2", adjust=False)}
    l14 = [c for c in t14 if c.split("-")[1][0] == "L"]
    res["H15_4_release_gate_O2"]["pooled_14_15"] = {
        d: ci(sum(ok(d, c, "O2") for c in lows) + sum(ok14(d, c, "O2") for c in l14), len(lows) + len(l14))
        for d in ("AAF-14", "B5", "B6", "AlwaysRollback", "L-Sec")}
    res["H15_4_release_gate_O2"]["feature_kept"] = {
        d: sum(1 for c in lows if (a := maj(dec[d].get(c) or [])) and table[c]["actions"][A.STATE_OF[a]]["delivered"]) for d in gate}

    # H15.5 security priority: cases whose current release carries a critical CVE or a secret
    cfg = {}
    for p in sorted((ROOT / "outcome_study15" / "measurements" / "rep1").glob("*.json")):
        d = json.loads(p.read_text()); cfg[d["case"]["id"]] = d["case"]
    secc = [c for c in faults if {"vuln", "secret"} & set(cfg[c]["current"])]
    res["H15_5_security"] = {"n": len(secc), "removed_O1": {d: sum(ok(d, c) for c in secc) for d in deciders},
                             "removed_any": {d: sum(1 for c in secc if (a := maj(dec[d].get(c) or []))
                                                    and not table[c]["actions"][A.STATE_OF[a]]["violations"]["security"]) for d in deciders}}
    res["H15_6_per_service"] = {s: {"H15_1": tests(ok, "AAF-14", SILOED, [c for c in faults if c.startswith(p)]),
                                    "H15_2": tests(ok, "AAF-14", ["B5", "B6", "AlwaysRollback"], [c for c in faults if c.startswith(p)])}
                                for s, p in SERVICES}
    res["H15_7_withheld"] = tests(ok, "AAF-14", ["B1-14", "B2-14", "B3-14"], withheld)

    # descriptive: discriminating cases, replicate agreement, LLM usage
    disc = [c for c in faults if len(table[c]["O1"]) < 5]
    res["discriminating"] = {"n": len(disc), "correct": {d: sum(ok(d, c) for c in disc) for d in deciders},
                             "H15_1": tests(ok, "AAF-14", SILOED, disc), "H15_2": tests(ok, "AAF-14", ["B5", "B6", "AlwaysRollback"], disc)}
    cal15 = json.loads((ROOT / "outcome_study15" / "calibration" / "calibration.json").read_text())
    p95_max = 2 * cal15["healthy_p95_median_ms"]

    def o1_single(d):  # O1 from one replicate, same violation rules as outcome_study/analyze.py
        nv = {}
        for a, m in d["outcomes"].items():
            p95, err = m["http"]["latency_p95_ms"], m["http"]["error_rate_pct"]
            nv[a] = (p95 is None or p95 > p95_max or err > 1.0) + (m["critical_cves_new"] > 0 or m["secret_in_config"]) + (m["footprint_cpu_m"] > 300.0)
        return sorted(a for a, v in nv.items() if v == min(nv.values()))
    reps = defaultdict(dict)
    for p in sorted((ROOT / "outcome_study15" / "measurements").glob("rep*/*.json")):
        d = json.loads(p.read_text()); reps[d["case"]["id"]][p.parent.name] = o1_single(d)
    both = [c for c, r in reps.items() if len(r) == 2]
    res["replicates"] = {"cases_with_two_replicates": len(both),
                         "O1_set_identical": sum(1 for c in both if list(reps[c].values())[0] == list(reps[c].values())[1]),
                         "differing": sorted(c for c in both if list(reps[c].values())[0] != list(reps[c].values())[1])}
    res["llm_usage_mean_per_variant"] = {d: {k: round(sum(v) / len(v), 2) for k, v in u.items() if v} for d, u in usage.items()}
    res["calibration"] = {k: v for k, v in json.loads((ROOT / "outcome_study15" / "calibration" / "calibration.json").read_text()).items()
                          if k in ("perf_cpu",)} | {"images": {k: v for k, v in json.loads(
                              (ROOT / "outcome_study15" / "calibration" / "calibration.json").read_text())["vuln_images"].items()
                              if k in ("G", "H", "NF", "P3", "P4", "NP", "C1", "C2", "NC")}}
    res["lowrisk_fallback"] = sorted(c for c, x in cfg.items() if x.get("lowrisk_fallback"))
    res["per_case"] = {c: {"O1": table[c.replace("-W", "")]["O1"], "O2": table[c.replace("-W", "")]["O2"],
                           **{d: maj(dec[d].get(c) or []) for d in deciders}} for c in cases + withheld}
    (ROOT / "outcome_study15" / "analysis.json").write_text(json.dumps(res, indent=2, default=str))
    print(json.dumps({k: v for k, v in res.items() if k not in ("per_case",)}, indent=1, default=str)[:15000])


if __name__ == "__main__":
    main()
