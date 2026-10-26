"""Analysis of the frontier-model deciders (PROTOCOL_GPT.md G1-G3). Written before any GPT run on outcome-study data."""
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

FAULT_GROUPS = ("R", "P", "S", "K", "D", "X")
STUDIES = ("outcome_study14", "outcome_study15", "outcome_study16")
GPT = ("AAF-14-GPT", "B5-GPT", "B6-GPT", "B3-GPT", "B3-14-GPT")


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
            if not r.get("error"):
                dec[r["decider"]][r["case"]].append(r["action"])
    g = here / "decisions_gpt" / "gpt_rows.jsonl"
    for line in (g.read_text().splitlines() if g.exists() else []):
        r = json.loads(line)
        if r.get("error"):
            continue
        dec[r["decider"]][r["case"]].append(r["action"])
        usage[r["decider"]]["calls"].append(r.get("n_calls") or 0)
        usage[r["decider"]]["tokens"].append((r.get("usage") or {}).get("total_tokens") or 0)
        if r["decider"] == "B5-GPT":
            dec["B3-GPT"][r["case"]].append(r["b3_action"]); dec["B3-14-GPT"][r["case"]].append(r["b3_14_action"])

    def maj(xs):
        c = Counter(xs).most_common()
        return sorted(a for a, n in c if n == c[0][1])[0] if c else None

    def ok(d, vid, oracle="O1"):
        a = maj(dec[d].get(vid) or [])
        return a is not None and A.STATE_OF.get(a) in table[vid.replace("-W", "")][oracle]
    return A, table, dec, ok, maj, usage


def tests(okf, a, others, ids, adjust=True):
    out = {b: mcnemar({c: okf(a, c) for c in ids}, {c: okf(b, c) for c in ids}, ids) for b in others}
    if adjust and len(out) > 1:
        for v, p in zip(out.values(), holm([v["p_exact"] for v in out.values()])):
            v["p_holm"] = p
    return out


def main() -> None:
    res, pooled = {"per_study": {}}, defaultdict(dict)
    for st in STUDIES:
        if not (ROOT / st / "decisions_gpt" / "gpt_rows.jsonl").exists():
            continue
        A, table, dec, ok, maj, usage = load(st)
        cases = sorted(table)
        faults = [c for c in cases if c.split("-")[1][0] in FAULT_GROUPS]
        allv = sorted({c for d in ("AAF-14", "AAF-14-GPT") for c in dec[d]})
        have = [c for c in faults if dec["B5-GPT"].get(c) and dec["B6-GPT"].get(c) and dec["AAF-14-GPT"].get(c)]
        withheld = [c for c in allv if c.endswith("-W") and dec["AAF-14-GPT"].get(c)]
        names = ["AAF-14", "AAF-14-GPT", "B5", "B6", "B5-GPT", "B6-GPT", "B3-GPT", "B3-14-GPT", "B1-14", "AlwaysRollback", "D-Sec"]
        r = {"n_faults": len(faults), "n_faults_with_gpt": len(have),
             "O1_faults": {d: sum(ok(d, c) for c in have) for d in names},
             "O2_faults": {d: sum(ok(d, c, "O2") for c in have) for d in names},
             "G1": tests(ok, "AAF-14", ["B5-GPT", "B6-GPT"], have, adjust=False),
             "G2_agreement": {"variants": len([c for c in allv if dec["AAF-14-GPT"].get(c) and dec["AAF-14"].get(c)]),
                              "same_action": sum(1 for c in allv if dec["AAF-14-GPT"].get(c) and dec["AAF-14"].get(c)
                                                 and maj(dec["AAF-14-GPT"][c]) == maj(dec["AAF-14"][c])),
                              "full_evidence_same": sum(1 for c in cases if dec["AAF-14-GPT"].get(c) and dec["AAF-14"].get(c)
                                                        and maj(dec["AAF-14-GPT"][c]) == maj(dec["AAF-14"][c])),
                              "full_evidence_n": sum(1 for c in cases if dec["AAF-14-GPT"].get(c) and dec["AAF-14"].get(c))},
             "G3_withheld": {"n": len(withheld), "correct": {d: sum(ok(d, c) for c in withheld) for d in ("AAF-14-GPT", "AAF-14", "B1-14", "B2-14", "B2", "B5-GPT", "B6-GPT") if dec[d]},
                             "tests": tests(ok, "AAF-14-GPT", [d for d in ("B1-14", "B2-14", "B2") if dec[d]], withheld, adjust=False)},
             "usage_mean_per_variant": {d: {k: round(sum(v) / len(v), 2) for k, v in u.items() if v} for d, u in usage.items()},
             "per_case": {c: {"O1": table[c.replace("-W", "")]["O1"], **{d: maj(dec[d].get(c) or []) for d in ("AAF-14", "AAF-14-GPT", "B5-GPT", "B6-GPT")}}
                          for c in allv}}
        res["per_study"][st] = r
        for c in have:
            for d in ("AAF-14", "B5-GPT", "B6-GPT", "AAF-14-GPT"):
                pooled[d][f"{st}:{c}"] = ok(d, c)
    ids = sorted(pooled["AAF-14"])
    if ids:
        res["G1_pooled"] = {"n": len(ids), "studies": sorted(res["per_study"]), "correct": {d: sum(v.values()) for d, v in pooled.items()},
                            "tests": tests(lambda d, c: pooled[d][c], "AAF-14", ["B5-GPT", "B6-GPT"], ids)}
    (ROOT / "outcome_study" / "analysis_gpt.json").write_text(json.dumps(res, indent=2, default=str))
    print(json.dumps({k: (v if k != "per_study" else {s: {kk: vv for kk, vv in x.items() if kk != "per_case"} for s, x in v.items()})
                      for k, v in res.items()}, indent=1, default=str)[:12000])


if __name__ == "__main__":
    main()
