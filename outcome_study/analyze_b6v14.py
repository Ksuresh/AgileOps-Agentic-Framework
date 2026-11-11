"""Analysis of the LLM decider given AAF's full v1.4 policy (PROTOCOL_B6V14.md). Written and committed before any B6-14 run.

Pre-specified (fault cases of Studies 14-16, primary oracle O1, majority over runs; pooled over studies):
  H2b-Q: AAF (the pre-specified Qwen3-8B investigator) vs B6-14 and B6-14E (Qwen3-8B); exact McNemar, Holm over 2.
  H2b-G: AAF vs B6-14-GPT and B6-14E-GPT (GPT-5.6 Luna); exact McNemar, Holm over 2.
  Family level (same design on different services of one study): sign test and cluster bootstrap, as for H1/H2.
Descriptive: per-study counts; O2; B6-14 vs the original B6 (effect of the policy) and B6-14E vs B6-14 (effect of the
evidence); agreement of each arm with the v1.4 rules' own action (rule-following); GPT-vs-Qwen agreement per arm;
withheld-evidence variants; calls, tokens and invalid outputs. A missing decision counts as incorrect."""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from outcome_study.analyze_gpt import load  # noqa: E402
from outcome_study.confirmatory_stats import compare, counts, family_level  # noqa: E402

STUDIES = ("outcome_study14", "outcome_study15", "outcome_study16")
FAULT = "RPSKDX"
NEW = {"Q": ["B6-14", "B6-14E"], "G": ["B6-14-GPT", "B6-14E-GPT"]}
OLD = ["B6", "B6-GPT", "B5", "B5-GPT"]
REF = ["AAF", "AAF-14-GPT", "B1-14"]


def gather():
    rows, wrows, extra = [], [], defaultdict(lambda: defaultdict(list))
    for st in STUDIES:
        A, table, dec, ok, maj, usage = load(st)
        for f in sorted((ROOT / st / "decisions_b6v14").glob("*.jsonl")):
            for line in f.read_text().splitlines():
                r = json.loads(line)
                if r.get("error"):
                    extra[r["decider"]]["errors"].append(1)
                    continue
                dec[r["decider"]][r["case"]].append(r["action"])
                extra[r["decider"]]["calls"].append(r.get("n_calls") or 0)
                extra[r["decider"]]["tokens"].append((r.get("usage") or {}).get("total_tokens") or 0)
                extra[r["decider"]]["invalid"].append(0 if (r.get("all_schema_valid") and r.get("all_evidence_refs_valid")) else 1)
        names = ["AAF" if d == "AAF-14" else d for d in ["AAF-14"] + REF[1:] + OLD + NEW["Q"] + NEW["G"]]
        src = {n: ("AAF-14" if n == "AAF" else n) for n in names}
        for c in sorted(table):
            g = c.split("-")[1][0]
            row = {"study": st, "case": c, "group": g, "family": f"{st}:{c.split('-')[1]}", "correct": {}, "correct_O2": {}, "action": {}}
            for n in names:
                d = src[n]
                if n in NEW["Q"] + NEW["G"] or dec[d].get(c):        # pre-specified: a missing new-arm decision counts as incorrect
                    row["correct"][n] = ok(d, c)
                    row["correct_O2"][n] = ok(d, c, "O2")
                    row["action"][n] = maj(dec[d].get(c) or [])
            (rows if g in FAULT else []).append(row)
            if g in FAULT and (dec["AAF-14"].get(c + "-W") or dec["AAF-14-GPT"].get(c + "-W")):
                w = {"case": c + "-W", "correct": {}}
                for n in names:
                    d = src[n]
                    if n in NEW["Q"] + NEW["G"] or dec[d].get(c + "-W"):
                        w["correct"][n] = ok(d, c + "-W")
                wrows.append(w)
    return rows, wrows, extra


def agree(rows, a, b):
    xs = [r for r in rows if r["action"].get(a) and r["action"].get(b)]
    return f"{sum(r['action'][a] == r['action'][b] for r in xs)}/{len(xs)}"


def main() -> None:
    rows, wrows, extra = gather()
    res = {"n_faults": len(rows), "per_study": {}, "pooled": {}}
    for scope in STUDIES + ("pooled",):
        R = rows if scope == "pooled" else [r for r in rows if r["study"] == scope]
        out = {"n": len(R), "O1": counts(R), "O2": counts(R, "correct_O2"),
               "H2b_Q": compare(R, "AAF", NEW["Q"]), "H2b_G": compare(R, "AAF", NEW["G"])}
        if scope == "pooled":
            out["family_level"] = {b: family_level(R, "AAF", b) for b in NEW["Q"] + NEW["G"]}
            out["policy_effect"] = {"B6-14 vs B6": compare(R, "B6-14", ["B6"], adjust=False),
                                    "B6-14-GPT vs B6-GPT": compare(R, "B6-14-GPT", ["B6-GPT"], adjust=False)}
            out["evidence_effect"] = {"B6-14E vs B6-14": compare(R, "B6-14E", ["B6-14"], adjust=False),
                                      "B6-14E-GPT vs B6-14-GPT": compare(R, "B6-14E-GPT", ["B6-14-GPT"], adjust=False)}
            out["follows_v14_rules"] = {d: agree(R, d, "B1-14") for d in NEW["Q"] + NEW["G"] + ["B6", "B6-GPT"]}
            out["gpt_vs_qwen_same_action"] = {"B6-14": agree(R, "B6-14", "B6-14-GPT"), "B6-14E": agree(R, "B6-14E", "B6-14E-GPT"),
                                              "B6": agree(R, "B6", "B6-GPT")}
        if scope == "pooled":
            res["pooled"] = out
        else:
            res["per_study"][scope] = out
    res["withheld"] = {"n": len(wrows), "O1": counts(wrows)}
    res["usage"] = {d: {"rows": len(v["calls"]), "errors": len(v["errors"]), "calls_mean": round(sum(v["calls"]) / max(1, len(v["calls"])), 2),
                        "tokens_total": sum(v["tokens"]), "invalid_outputs": sum(v["invalid"])} for d, v in extra.items()}
    (ROOT / "outcome_study" / "analysis_b6v14.json").write_text(json.dumps(res, indent=2, default=str))
    p = res["pooled"]
    print(json.dumps({"O1": {k: p["O1"].get(k) for k in ["AAF", "B1-14", "B6", "B6-GPT"] + NEW["Q"] + NEW["G"]},
                      "O2": {k: p["O2"].get(k) for k in ["AAF"] + NEW["Q"] + NEW["G"]},
                      "H2b_Q": p["H2b_Q"], "H2b_G": p["H2b_G"], "follows_v14": p["follows_v14_rules"]}, indent=1, default=str))


if __name__ == "__main__":
    main()
