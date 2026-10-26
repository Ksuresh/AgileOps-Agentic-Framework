"""All numbers reported for the confirmatory outcome studies (14-16), computed from committed artifacts only.

Units of analysis (stated so that no test inflates the sample):
  * case      one incident definition on one service (the unit of every McNemar test, as pre-specified);
  * family    the same case design on different services of one study (e.g. UF-R01, UP-R01, UC-R01); cases in a family
              share a fault recipe, so they are not fully independent;
  * variant   the withheld-evidence copy of a case: analysed separately, never pooled with full-evidence cases;
  * replicate the two measurements of a case: averaged before the oracle, never counted as separate observations;
  * LLM run   repeated runs of a configuration: reduced to the majority action before testing.
Besides the pre-specified case-level tests, this script reports family-level sign tests and family-cluster bootstrap
intervals, so that conclusions can be checked at the coarser unit.

Subsets: all fault cases; discriminating cases (not every action is O1-correct); unique-answer cases (exactly one action is
O1-correct), where the actions have clearly different consequences.
Output: outcome_study/confirmatory_stats.json."""
from __future__ import annotations

import json
import random
import subprocess
import sys
from collections import defaultdict
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from external_validation.expert_study.analyze import holm, mcnemar, wilson  # noqa: E402
from outcome_study.analyze_gpt import load  # noqa: E402

STUDIES = ("outcome_study14", "outcome_study15", "outcome_study16")
FAULT = "RPSKDX"
SILOED = ["D-DevOps", "D-SRE", "D-FinOps", "D-Sec", "L-DevOps", "L-SRE", "L-FinOps", "L-Sec"]
LLM = ["B5", "B6", "B5-GPT", "B6-GPT"]
POLICIES = ["AlwaysRollback", "Dominant", "BlockAnyCVE", "AlwaysNone"]
OTHERS = ["B1-14", "B2-14", "B1", "B1-12", "AAF-14-GPT", "B3", "B3-14", "B3-GPT", "B3-14-GPT"]


def sign_test(wins: int, losses: int) -> float:
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def gather():
    data = {}  # (study, case) -> dict
    for st in STUDIES:
        A, table, dec, ok, maj, usage = load(st)
        siloed14 = {d: d + "-14" if (d + "-14") in dec else d for d in SILOED}
        for c in sorted(table):
            g = c.split("-")[1][0]
            row = {"study": st, "case": c, "group": g, "family": f"{st}:{c.split('-')[1]}", "O1": table[c]["O1"],
                   "O2": table[c]["O2"], "n_reps": table[c]["n_reps"], "correct": {}, "correct_O2": {}, "action": {}}
            names = ["AAF-14"] + list(siloed14.values()) + LLM + POLICIES + OTHERS + [d for d in SILOED if d.startswith("D-")]
            for d in dict.fromkeys(names):
                if dec[d].get(c):
                    key = "AAF" if d == "AAF-14" else d
                    row["correct"][key] = ok(d, c)
                    row["correct_O2"][key] = ok(d, c, "O2")
                    row["action"][key] = maj(dec[d][c])
            for d, src in siloed14.items():  # primary siloed comparison: v1.4 rules where the study has them
                if dec[src].get(c):
                    row["correct"]["S:" + d] = ok(src, c)
                    row["correct_O2"]["S:" + d] = ok(src, c, "O2")
            data[(st, c)] = row
            if dec["AAF-14"].get(c + "-W") or dec["AAF-14-GPT"].get(c + "-W"):
                w = {"study": st, "case": c + "-W", "group": g, "family": row["family"], "O1": table[c]["O1"], "correct": {}}
                for d in ("AAF-14", "AAF-14-GPT", "B1-14", "B2-14", "B2", "B1", "B5", "B6", "B5-GPT", "B6-GPT", "B3-14"):
                    if dec[d].get(c + "-W"):
                        w["correct"]["AAF" if d == "AAF-14" else d] = ok(d, c + "-W")
                data[(st, c + "-W")] = w
    return data


def compare(rows, a, others, adjust=True):
    ids = [r for r in rows if a in r["correct"] and all(b in r["correct"] for b in others)]
    out = {}
    for b in others:
        idx = list(range(len(ids)))
        out[b] = mcnemar({i: ids[i]["correct"][a] for i in idx}, {i: ids[i]["correct"][b] for i in idx}, idx)
        out[b]["n"] = len(ids)
    if adjust and len(out) > 1:
        for v, p in zip(out.values(), holm([v["p_exact"] for v in out.values()])):
            v["p_holm"] = p
    return out


def family_level(rows, a, b, boot=4000, seed=15):
    fam = defaultdict(lambda: [0, 0, 0])  # family -> [a correct, b correct, n]
    for r in rows:
        if a in r["correct"] and b in r["correct"]:
            f = fam[r["family"]]
            f[0] += r["correct"][a]; f[1] += r["correct"][b]; f[2] += 1
    fams = list(fam.values())
    wins = sum(1 for x in fams if x[0] > x[1]); losses = sum(1 for x in fams if x[0] < x[1])
    rng = random.Random(seed)
    diffs = []
    for _ in range(boot):
        s = [fams[rng.randrange(len(fams))] for _ in fams]
        n = sum(x[2] for x in s)
        diffs.append((sum(x[0] for x in s) - sum(x[1] for x in s)) / n)
    diffs.sort()
    return {"families": len(fams), "families_a_better": wins, "families_b_better": losses, "sign_test_p": sign_test(wins, losses),
            "acc_diff": round((sum(x[0] for x in fams) - sum(x[1] for x in fams)) / sum(x[2] for x in fams), 3),
            "cluster_bootstrap_95": [round(diffs[int(0.025 * boot)], 3), round(diffs[int(0.975 * boot) - 1], 3)]}


def counts(rows, key="correct"):
    names = sorted({d for r in rows for d in r[key]})
    return {d: f"{sum(r[key][d] for r in rows if d in r[key])}/{sum(1 for r in rows if d in r[key])}" for d in names}


def timeline():
    def first(path, grep=None):
        cmd = ["git", "log", "--reverse", "--format=%h %cI %s", "--", path]
        out = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True).stdout.splitlines()
        return out[0] if out else None
    return {"rules_v14_frozen": first("agentic_experiments/rules_v14.py"),
            **{f"{st}_protocol": first(f"{st}/PROTOCOL.md") for st in STUDIES},
            **{f"{st}_first_measurement": first(f"{st}/measurements") for st in STUDIES},
            "rules_v14_last_change": subprocess.run(["git", "log", "-1", "--format=%h %cI %s", "--", "agentic_experiments/rules_v14.py"],
                                                    cwd=ROOT, capture_output=True, text=True).stdout.strip()}


def main() -> None:
    data = gather()
    res = {"timeline": timeline(), "per_study": {}, "pooled": {}}
    full = [r for r in data.values() if not r["case"].endswith("-W") and r["group"] in FAULT]
    low = [r for r in data.values() if not r["case"].endswith("-W") and r["group"] == "L"]
    wh = [r for r in data.values() if r["case"].endswith("-W")]
    sil = ["S:" + d for d in SILOED]
    for scope, sel in [(st, lambda r, st=st: r["study"] == st) for st in STUDIES] + [("pooled", lambda r: True)]:
        F = [r for r in full if sel(r)]
        disc = [r for r in F if len(r["O1"]) < 5]
        uniq = [r for r in F if len(r["O1"]) == 1]
        out = {"n_faults": len(F), "n_families": len({r["family"] for r in F}), "n_discriminating": len(disc), "n_unique_answer": len(uniq),
               "O1": counts(F), "O2": counts(F, "correct_O2"), "O1_discriminating": counts(disc), "O1_unique_answer": counts(uniq),
               "H1_case_level": compare(F, "AAF", [s for s in sil if all(s in r["correct"] for r in F)]),
               "H2_case_level": compare(F, "AAF", [b for b in LLM + ["AlwaysRollback"] if any(b in r["correct"] for r in F)]),
               "H1_discriminating": compare(disc, "AAF", [s for s in sil if all(s in r["correct"] for r in disc)]),
               "H2_unique_answer": compare(uniq, "AAF", [b for b in LLM + ["AlwaysRollback"] if any(b in r["correct"] for r in uniq)], adjust=False),
               "family_level": {b: family_level(F, "AAF", b) for b in sil + LLM + ["AlwaysRollback", "Dominant"] if any(b in r["correct"] for r in F)}}
        L = [r for r in low if sel(r)]
        out["H3_low_risk"] = {"n": len(L), "O2": {d: dict(zip(("k", "n"), map(int, v.split("/"))), wilson95_pct=wilson(*map(int, v.split("/"))))
                                                  for d, v in counts(L, "correct_O2").items()},
                              # release continued = the action keeps the feature delivered (every action except rollback in this testbed)
                              "release_continued": {d: f"{sum(1 for r in L if r['action'].get(d) and r['action'][d] != 'Rollback to stable deployment')}/{sum(1 for r in L if r['action'].get(d))}"
                                                    for d in ("AAF", "AAF-14-GPT", "B5", "B6", "B5-GPT", "B6-GPT", "AlwaysRollback", "BlockAnyCVE", "AlwaysNone")}}
        W = [r for r in wh if sel(r)]
        out["withheld"] = {"n": len(W), "O1": counts(W), "AAF_vs": compare(W, "AAF", [b for b in ("B1-14", "B2-14") if all(b in r["correct"] for r in W)], adjust=False),
                           "AAF_GPT_vs": compare(W, "AAF-14-GPT", [b for b in ("B1-14", "B2-14") if all(b in r["correct"] for r in W)], adjust=False)}
        (res["per_study"].__setitem__(scope, out) if scope != "pooled" else res.__setitem__("pooled", out))
    allc = [r for r in data.values() if not r["case"].endswith("-W")]
    res["model_independence_outcome"] = {"cases": sum(1 for r in allc if "AAF" in r["action"] and "AAF-14-GPT" in r["action"]),
                                         "same_action_qwen_vs_gpt_investigator": sum(1 for r in allc if r["action"].get("AAF") and r["action"].get("AAF") == r["action"].get("AAF-14-GPT")),
                                         "same_action_AAF_vs_rules_only": sum(1 for r in allc if r["action"].get("AAF") and r["action"].get("AAF") == r["action"].get("B1-14"))}
    (ROOT / "outcome_study" / "confirmatory_stats.json").write_text(json.dumps(res, indent=2, default=str))
    p = res["pooled"]
    print(json.dumps({"timeline": res["timeline"], "pooled_O1": p["O1"], "n": [p["n_faults"], p["n_families"], p["n_discriminating"], p["n_unique_answer"]],
                      "unique": p["O1_unique_answer"], "family": {k: (v["families_a_better"], v["families_b_better"], v["sign_test_p"], v["cluster_bootstrap_95"]) for k, v in p["family_level"].items()},
                      "mi": res["model_independence_outcome"]}, indent=1, default=str))


if __name__ == "__main__":
    main()
