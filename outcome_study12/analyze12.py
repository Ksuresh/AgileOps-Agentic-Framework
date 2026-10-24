"""Study 12 analysis (PROTOCOL.md): H12.1 AAF-12 vs AAF; H12.2 vs siloed experts; H12.3 vs B5, B6, AlwaysRollback;
H12.4 predating-fault cases; withheld variants secondary. Uses outcome_study.analyze for the oracle."""
from __future__ import annotations

import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault("OUTCOME_DIR", "outcome_study12")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from outcome_study import analyze as A  # noqa: E402
from external_validation.expert_study.analyze import holm, mcnemar  # noqa: E402

HERE = ROOT / "outcome_study12"


def main() -> None:
    table = A.outcome_table(json.loads((HERE / "calibration" / "calibration.json").read_text()))
    dec = defaultdict(lambda: defaultdict(list))
    for line in (HERE / "decisions" / "deterministic.jsonl").read_text().splitlines():
        r = json.loads(line); dec[r["decider"]][r["case"]].append(r["action"])
    errors = Counter()
    for f in sorted((HERE / "decisions").glob("llm_shard_*.jsonl")):
        for line in f.read_text().splitlines():
            r = json.loads(line)
            if r.get("error"):
                errors[r["decider"]] += 1; continue
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

    cases = sorted(table)
    faults = [c for c in cases if not c.startswith("PC")]
    pp = [c for c in cases if c.startswith("PP")]
    withheld = [c + "-W" for c in cases if (c + "-W") in dec["B1"]]
    deciders = sorted(dec)
    res = {"errors": dict(errors), "runs": {d: Counter(len(v) for v in dec[d].values()) for d in ("AAF", "AAF-12", "B5", "B6")},
           "n_faults": len(faults), "n_predating": len(pp), "n_withheld": len(withheld)}
    for oracle in ("O1", "O2", "O3"):
        res[f"accuracy_{oracle}"] = {g: {d: sum(ok(d, c, oracle) for c in ids) for d in deciders} | {"n": len(ids)}
                                     for g, ids in (("faults", faults), ("predating", pp), ("controls", [c for c in cases if c.startswith("PC")]), ("withheld", withheld))}

    def tests(a, others, ids, oracle="O1", adjust=True):
        out = {b: mcnemar({c: ok(a, c, oracle) for c in ids}, {c: ok(b, c, oracle) for c in ids}, ids) for b in others}
        if adjust:
            for (b, v), p in zip(out.items(), holm([v["p_exact"] for v in out.values()])):
                v["p_holm"] = p
        return out
    res["H12_1_AAF12_vs_AAF"] = tests("AAF-12", ["AAF"], faults, adjust=False)
    res["H12_2_vs_siloed"] = tests("AAF-12", ["D-DevOps", "D-SRE", "D-FinOps", "D-Sec", "L-DevOps", "L-SRE", "L-FinOps", "L-Sec"], faults)
    res["H12_2b_original_AAF_vs_siloed"] = tests("AAF", ["D-DevOps", "D-SRE", "D-FinOps", "D-Sec", "L-DevOps", "L-SRE", "L-FinOps", "L-Sec"], faults)
    res["H12_3_vs_LLM_and_policies"] = tests("AAF-12", ["B5", "B6", "AlwaysRollback"], faults)
    res["H12_4_predating"] = {d: sum(ok(d, c) for c in pp) for d in ("AAF-12", "AAF", "B1-12", "B1", "AlwaysRollback")}
    res["withheld_AAF12_vs"] = tests("AAF-12", ["B1-12", "B2", "AAF"], withheld)
    res["per_case"] = {c: {"O1": table[c.replace("-W", "")]["O1"], **{d: maj(dec[d].get(c) or []) for d in deciders}} for c in cases + withheld}
    (HERE / "analysis.json").write_text(json.dumps(res, indent=2, default=str))
    print(json.dumps({k: v for k, v in res.items() if k != "per_case"}, indent=1, default=str))


if __name__ == "__main__":
    main()
