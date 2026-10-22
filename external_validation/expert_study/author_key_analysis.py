"""Primary analysis of the new cases on the sealed author key (amendment A2). Same arms and tests as analyze.py."""
import json, sys
from pathlib import Path
import yaml
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from external_validation.expert_study.analyze import mcnemar, holm, wilson  # noqa: E402
HERE = Path(__file__).resolve().parent
d = json.loads((HERE / "analysis.json").read_text())
A = yaml.safe_load((HERE / "author_labels.yaml").read_text())
new = {c["id"]: c for c in yaml.safe_load((HERE / "cases_new_v1.yaml").read_text())["cases"]}
pc = d["per_case_new"]; CON = {"NV-U19", "NV-U20", "NV-U21", "NV-U24"}
ARMS = ("DET", "AF-M", "AF-MS", "FETCH-ALL", "HY", "B3", "AO", "B6")
maj = lambda acts, s: bool(acts) and sum(a in s for a in acts) * 2 > len(acts)
M = {a: {i: maj(pc[i][a], A[i]) for i in new} for a in ARMS}
groups = {"standard": [i for i in new if new[i]["stratum"] != "UNMARKED"],
          "unmarked_decisive": [i for i in new if new[i]["stratum"] == "UNMARKED" and i not in CON],
          "unmarked_controls": sorted(CON)}
for sub in ("absent", "stale", "cross_domain"):
    groups["unmarked_" + sub] = [i for i in groups["unmarked_decisive"] if new[i]["substratum"] == sub]
for st in ("CLEAR", "AMBIGUOUS", "INCOMPLETE", "MISLEADING"):
    groups["standard_" + st] = [i for i in new if new[i]["stratum"] == st]
res = {"key": "author (sealed before labelling and runs)",
       "groups": {g: {"n": len(ids), **{a: {"correct": sum(M[a][i] for i in ids), "wilson": wilson(sum(M[a][i] for i in ids), len(ids)),
                                            "tool_calls_per_case": d["new_cases"].get(g, {}).get(a, {}).get("tool_calls_per_case")} for a in ARMS}}
                  for g, ids in groups.items()}}
U, S = groups["unmarked_decisive"], groups["standard"]
t1, t2 = mcnemar(M["HY"], M["AF-MS"], U), mcnemar(M["HY"], M["AF-M"], U); adj = holm([t1["p_exact"], t2["p_exact"]])
res["H3"] = {"HY_vs_AF-MS": {**t1, "p_holm": adj[0]}, "HY_vs_AF-M": {**t2, "p_holm": adj[1]},
             "HY_vs_FETCH-ALL": mcnemar(M["HY"], M["FETCH-ALL"], U), "B6_vs_AF-MS": mcnemar(M["B6"], M["AF-MS"], U),
             "B6_vs_FETCH-ALL": mcnemar(M["B6"], M["FETCH-ALL"], U), "B6_vs_HY": mcnemar(M["B6"], M["HY"], U)}
res["H4_new_standard"] = {"HY_vs_DET": mcnemar(M["HY"], M["DET"], S), "HY_vs_B6": mcnemar(M["HY"], M["B6"], S), "HY_vs_AO": mcnemar(M["HY"], M["AO"], S)}
# pooled with original 48 on the author key (original majority-of-runs from the revision run)
import csv
from collections import defaultdict
orig = {c["id"]: c for f in ("cases_v1.yaml", "cases_replication_v1.yaml") for c in yaml.safe_load((HERE.parents[1] / "agentic_experiments" / f).read_text())["cases"]}
acts = defaultdict(lambda: defaultdict(list))
for r in csv.DictReader((HERE.parents[1] / "revision_2026/results/llm_arms/20261022T064831Z/all_runs.csv").open()):
    if r["action"]:
        acts[r["arm"]][r["case"]].append(r["action"])
from agentic_experiments.hybrid import deterministic_assessment
import copy
for cid, c in orig.items():
    acts["DET"][cid] = [deterministic_assessment(copy.deepcopy(c["evidence"]))["selected_action"]]
P = {a: {**{cid: maj(acts[a][cid], orig[cid]["admissible_actions"]) for cid in orig}, **{i: M[a][i] for i in S}} for a in ("DET", "HY", "B6", "AO")}
ids = list(orig) + S
res["H4_pooled_72"] = {"n": len(ids), **{a: sum(P[a].values()) for a in P}, "HY_vs_DET": mcnemar(P["HY"], P["DET"], ids),
                       "HY_vs_B6": mcnemar(P["HY"], P["B6"], ids), "HY_vs_AO": mcnemar(P["HY"], P["AO"], ids)}
res["HY_trigger_rate"] = d["HY_trigger_rate"]; res["HY_same_action_all_3_runs"] = d["HY_stability_new"]; res["budget"] = d["budget"]
res["practitioner_ai_assisted_sensitivity"] = {"H1": {k: {x: v[x] for x in v if x != "disagreements"} for k, v in d["H1"].items()},
                                               "H2_original48": d["H2_original48"], "new_cases": {g: {a: v[a]["correct"] for a in ARMS} for g, v in d["new_cases"].items()}}
(HERE / "analysis_author_key.json").write_text(json.dumps(res, indent=1))
print(json.dumps({k: v for k, v in res.items() if k != "practitioner_ai_assisted_sensitivity"}, indent=0)[:6000])
