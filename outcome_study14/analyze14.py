"""Study 14 analysis (PROTOCOL.md H14.1-H14.7) plus per-service breakdown and pooled original-AAF analysis."""
from __future__ import annotations

import importlib
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from external_validation.expert_study.analyze import holm, mcnemar  # noqa: E402

SILOED = ["D-DevOps", "D-SRE", "D-FinOps", "D-Sec", "L-DevOps", "L-SRE", "L-FinOps", "L-Sec"]
FAULT_GROUPS = ("R", "P", "S", "K", "D", "X")


def load(study: str):
    os.environ["OUTCOME_DIR"] = study
    from outcome_study import analyze as A
    importlib.reload(A)
    here = ROOT / study
    table = A.outcome_table(json.loads((here / "calibration" / "calibration.json").read_text()))
    dec = defaultdict(lambda: defaultdict(list))
    for line in (here / "decisions" / "deterministic.jsonl").read_text().splitlines():
        r = json.loads(line); dec[r["decider"]][r["case"]].append(r["action"])
    for f in sorted((here / "decisions").glob("llm_shard_*.jsonl")):
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

    def ok(d, vid, oracle="O1"):
        a = maj(dec[d].get(vid) or [])
        return a is not None and A.STATE_OF.get(a) in table[vid.replace("-W", "")][oracle]
    return table, dec, ok, maj


def tests(ok, a, others, ids, oracle="O1", adjust=True):
    out = {b: mcnemar({c: ok(a, c, oracle) for c in ids}, {c: ok(b, c, oracle) for c in ids}, ids) for b in others}
    if adjust and len(out) > 1:
        for (b, v), p in zip(out.items(), holm([v["p_exact"] for v in out.values()])):
            v["p_holm"] = p
    return out


def main() -> None:
    table, dec, ok, maj = load("outcome_study14")
    cases = sorted(table)
    group = lambda c: c.split("-")[1][0]
    faults = [c for c in cases if group(c) in FAULT_GROUPS]
    traps = [c for c in cases if group(c) == "P"]
    withheld = [c + "-W" for c in cases if (c + "-W") in dec["B1"]]
    deciders = sorted(dec)
    res = {"n_cases": len(cases), "n_faults": len(faults), "n_withheld": len(withheld),
           "runs": {d: dict(Counter(len(v) for v in dec[d].values())) for d in ("AAF", "AAF-14", "B5", "B6")}}
    for oracle in ("O1", "O2", "O3"):
        res[f"accuracy_{oracle}"] = {}
        for g, ids in (("faults", faults), ("frontend_faults", [c for c in faults if c.startswith("QF")]),
                       ("productcatalog_faults", [c for c in faults if c.startswith("QP")]),
                       *[(f"group_{x}", [c for c in cases if group(c) == x]) for x in ("C", "L", "R", "P", "S", "K", "D", "X")],
                       ("withheld", withheld)):
            res[f"accuracy_{oracle}"][g] = {d: sum(ok(d, c, oracle) for c in ids) for d in deciders} | {"n": len(ids)}
    res["H14_1_vs_siloed"] = tests(ok, "AAF-14", SILOED, faults)
    res["H14_2_vs_AAF_original"] = tests(ok, "AAF-14", ["AAF"], faults, adjust=False)
    res["H14_3_vs_LLM_and_policies"] = tests(ok, "AAF-14", ["B5", "B6", "AlwaysRollback"], faults)
    res["H14_4_traps"] = {"counts": {d: sum(ok(d, c) for c in traps) for d in ("AAF-14", "AAF", "B1-14", "B1-12", "B1", "AlwaysRollback")},
                          "tests": tests(ok, "B1-14", ["B1-12"], traps, adjust=False) | tests(ok, "AAF-14", ["AAF"], traps, adjust=False)}
    sec = {"K_S_critical_removed_O1": {}, "L_release_continued_O2": {}}
    for d in deciders:
        sec["K_S_critical_removed_O1"][d] = f"{sum(ok(d, c) for c in cases if group(c) in 'KS')}/{sum(1 for c in cases if group(c) in 'KS')}"
        sec["L_release_continued_O2"][d] = f"{sum(ok(d, c, 'O2') for c in cases if group(c) == 'L')}/{sum(1 for c in cases if group(c) == 'L')}"
    res["H14_5_security"] = sec
    res["H14_6_per_service"] = {svc: {"H14_1": tests(ok, "AAF-14", SILOED, [c for c in faults if c.startswith(p)]),
                                      "H14_3": tests(ok, "AAF-14", ["B5", "B6", "AlwaysRollback"], [c for c in faults if c.startswith(p)])}
                                for svc, p in (("frontend", "QF"), ("productcatalogservice", "QP"))}
    res["withheld_AAF14_vs"] = tests(ok, "AAF-14", ["B1-14", "B2", "AAF"], withheld)
    # H14.7 pooled original AAF (rule set v1.0, frozen since Study 9) vs siloed rule experts over Studies 9, 12, 14
    pooled_ok = defaultdict(dict)
    for study, is_fault in (("outcome_study", lambda c: c.startswith(("OS", "OX"))), ("outcome_study12", lambda c: not c.startswith("PC")),
                            ("outcome_study14", lambda c: group(c) in FAULT_GROUPS)):
        t, d_, ok_, _ = load(study)
        for c in t:
            if is_fault(c):
                for d in ("AAF", "D-DevOps", "D-SRE", "D-FinOps", "D-Sec", "B5", "B6"):
                    if d in d_:
                        pooled_ok[d][f"{study}:{c}"] = ok_(d, c)
    ids = sorted(pooled_ok["AAF"])
    res["H14_7_pooled"] = {"n": len(ids), "correct": {d: sum(v.values()) for d, v in pooled_ok.items()},
                           "tests": tests(lambda d, c, o="O1": pooled_ok[d][c], "AAF", ["D-DevOps", "D-SRE", "D-FinOps", "D-Sec"], ids)}
    res["per_case"] = {c: {"O1": table[c.replace("-W", "")]["O1"], "O2": table[c.replace("-W", "")]["O2"], **{d: maj(dec[d].get(c) or []) for d in deciders}}
                       for c in cases + withheld}
    (ROOT / "outcome_study14" / "analysis.json").write_text(json.dumps(res, indent=2, default=str))
    print(json.dumps({k: v for k, v in res.items() if k not in ("per_case",)}, indent=1, default=str)[:12000])


if __name__ == "__main__":
    main()
