"""Scores the expert-labelled extension study exactly as pre-specified in PROTOCOL.md (H1-H4).

Inputs: expert_labels.json (code -> label), label_map.json (code -> case), author keys (original YAML admissible
actions; author_labels.yaml for new cases), original-48 LLM outputs (revision_2026/results/llm_arms/<run>/all_runs.csv),
new-case LLM outputs (results/<run>/{HY,AO,B6}.jsonl). Deterministic arms are recomputed here.
Usage: python analyze.py <new-case results dir>
"""
from __future__ import annotations

import copy
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from agentic_experiments.agentic_runtime import ACTIONS, merge_tool_result  # noqa: E402
from agentic_experiments.hybrid import deterministic_assessment, missing_fields  # noqa: E402
from revision_2026.autofetch_baseline import FIELD_TO_TOOL  # noqa: E402

ORIG_RUNS = ROOT / "revision_2026/results/llm_arms/20261022T064831Z/all_runs.csv"
CONTROLS = {"NV-U19", "NV-U20", "NV-U21", "NV-U24"}


# ---------------- statistics ----------------
def mcnemar(a: dict, b: dict, ids) -> dict:
    n10 = sum(1 for i in ids if a[i] and not b[i]); n01 = sum(1 for i in ids if b[i] and not a[i]); d = n10 + n01
    p = 1.0 if d == 0 else min(1.0, 2 * sum(math.comb(d, k) for k in range(0, min(n10, n01) + 1)) / 2 ** d)
    return {"a_only": n10, "b_only": n01, "p_exact": p}


def holm(ps: list[float]) -> list[float]:
    order = sorted(range(len(ps)), key=lambda i: ps[i]); adj = [0.0] * len(ps); run = 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - r) * ps[i])); adj[i] = run
    return adj


def wilson(k: int, n: int) -> list[float]:
    if n == 0:
        return [0.0, 0.0]
    z = 1.96; p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(100 * (c - h), 1), round(100 * (c + h), 1)]


def kappa(x: list, y: list) -> float:
    n = len(x); po = sum(a == b for a, b in zip(x, y)) / n
    cx, cy = Counter(x), Counter(y); pe = sum(cx[k] * cy[k] for k in set(cx) | set(cy)) / n ** 2
    return round((po - pe) / (1 - pe), 3) if pe < 1 else 1.0


# ---------------- deterministic arms ----------------
def tool_exec(case, tool):
    return copy.deepcopy(case["tool_result"]) if tool == case.get("expected_tool") else None


def det_arm(case, mode):
    ev = copy.deepcopy(case["evidence"]); tools = list(case.get("available_tools") or []); calls = []
    if mode == "DET":
        pass
    elif mode in ("AF-M", "AF-MS"):
        fields = list(missing_fields(ev))
        if mode == "AF-MS":
            for dom, block in ev.items():
                for f, meta in ((block or {}).get("_evidence") or {}).items():
                    if isinstance(meta, dict) and meta.get("status") == "stale":
                        fields.append(f"{dom}.{f}")
        for f in fields:
            t = FIELD_TO_TOOL.get(f)
            if t and t in tools and t not in calls:
                calls.append(t)
    elif mode == "FETCH-ALL":
        calls = tools[:]
    for t in calls:
        r = tool_exec(case, t)
        if r:
            ev = merge_tool_result(ev, r)
    return str(deterministic_assessment(ev)["selected_action"]), len(calls)


def main() -> None:
    newdir = Path(sys.argv[1]) if len(sys.argv) > 1 else sorted((HERE / "results").glob("2026*"))[-1]
    labels = json.loads((HERE / "expert_labels.json").read_text())
    cmap = json.loads((HERE / "label_map.json").read_text())
    orig = {c["id"]: c for f in ("cases_v1.yaml", "cases_replication_v1.yaml") for c in yaml.safe_load((ROOT / "agentic_experiments" / f).read_text())["cases"]}
    new = {c["id"]: c for c in yaml.safe_load((HERE / "cases_new_v1.yaml").read_text())["cases"]}
    author_new = yaml.safe_load((HERE / "author_labels.yaml").read_text())
    author = {**{k: c["admissible_actions"] for k, c in orig.items()}, **author_new}
    expert, cannot = {}, []
    for code, rec in labels.items():
        cid = cmap[code]["case"]
        if rec.get("cannot"):
            cannot.append(cid); continue
        expert[cid] = {"set": list(rec["acceptable"]), "best": rec["best"], "confidence": rec.get("confidence")}
    res = {"n_labelled": len(labels), "cannot_decide": sorted(cannot), "n_scored": len(expert)}

    # H1 oracle agreement
    def h1(ids):
        ids = [i for i in ids if i in expert]
        best_in = sum(expert[i]["best"] in author[i] for i in ids)
        jac = [len(set(expert[i]["set"]) & set(author[i])) / len(set(expert[i]["set"]) | set(author[i])) for i in ids]
        return {"n": len(ids), "expert_best_in_author_set": best_in, "pct": round(100 * best_in / max(1, len(ids)), 1),
                "wilson": wilson(best_in, len(ids)), "mean_jaccard": round(sum(jac) / max(1, len(jac)), 3),
                "kappa_best_vs_author_first": kappa([expert[i]["best"] for i in ids], [author[i][0] for i in ids]),
                "disagreements": [{"case": i, "expert_best": expert[i]["best"], "expert_set": expert[i]["set"], "author": author[i]}
                                  for i in ids if expert[i]["best"] not in author[i]]}
    res["H1"] = {"all": h1(list(orig) + list(new)), "original": h1(list(orig)), "new": h1(list(new))}

    def ok(cid, action, key="expert"):
        s = expert[cid]["set"] if key == "expert" else author[cid]
        return action in s

    # H2 original 48 under expert key
    rows = list(csv.DictReader(ORIG_RUNS.open()))
    acts = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["action"]:
            acts[r["arm"]][r["case"]].append(r["action"])
    for arm in ("DET", "AF-M"):
        for cid, c in orig.items():
            acts[arm][cid] = [det_arm(c, arm)[0]]

    def majority(arm, cid, key):
        a = acts[arm][cid]
        return sum(ok(cid, x, key) for x in a) * 2 > len(a)
    o_ids = [i for i in orig if i in expert]
    h2 = {}
    for arm in ("DET", "AF-M", "HY", "B3", "AO", "B6", "PV1", "PV2"):
        e = {i: majority(arm, i, "expert") for i in o_ids}; a = {i: majority(arm, i, "author") for i in o_ids}
        h2[arm] = {"expert_key": sum(e.values()), "author_key": sum(a.values()), "n": len(o_ids), "wilson_expert": wilson(sum(e.values()), len(o_ids))}
    E = {arm: {i: majority(arm, i, "expert") for i in o_ids} for arm in ("DET", "HY", "B6", "AO", "AF-M")}
    h2["tests"] = {"HY_vs_DET": mcnemar(E["HY"], E["DET"], o_ids), "HY_vs_B6": mcnemar(E["HY"], E["B6"], o_ids),
                   "HY_vs_AO": mcnemar(E["HY"], E["AO"], o_ids), "HY_vs_AF-M": mcnemar(E["HY"], E["AF-M"], o_ids)}
    res["H2_original48"] = h2

    # New cases: all arms
    nacts, ntools, ninv, nerr = defaultdict(lambda: defaultdict(list)), defaultdict(dict), {}, Counter()
    for cid, c in new.items():
        for arm in ("DET", "AF-M", "AF-MS", "FETCH-ALL"):
            a, k = det_arm(c, arm); nacts[arm][cid] = [a]; ntools[arm][cid] = k
    for arm in ("HY", "AO", "B6"):
        f = newdir / f"{arm}.jsonl"
        for line in f.read_text().splitlines() if f.exists() else []:
            r = json.loads(line)
            if r.get("error"):
                nerr[arm] += 1; continue
            nacts[arm][r["case"]].append(r["action"])
            ntools[arm].setdefault(r["case"], []).append(len(r.get("tool_events") or []))
            if arm == "HY":
                ninv.setdefault(r["case"], []).append(bool(r.get("agentic_invoked")))
            if arm == "AO":  # B3: always-on agents + deterministic authority
                ev = copy.deepcopy(new[r["case"]]["evidence"])
                for t in r.get("tool_events") or []:
                    if t.get("tool") == new[r["case"]].get("expected_tool") and isinstance(t.get("result"), dict):
                        ev = merge_tool_result(ev, t["result"])
                nacts["B3"][r["case"]].append(str(deterministic_assessment(ev)["selected_action"]))
    n_ids = [i for i in new if i in expert]

    def nmaj(arm, cid):
        a = nacts[arm].get(cid) or []
        return bool(a) and sum(ok(cid, x) for x in a) * 2 > len(a)
    groups = {"standard": [i for i in n_ids if new[i]["stratum"] != "UNMARKED"],
              "unmarked_decisive": [i for i in n_ids if new[i]["stratum"] == "UNMARKED" and i not in CONTROLS],
              "unmarked_controls": [i for i in n_ids if i in CONTROLS]}
    for sub in ("absent", "stale", "cross_domain"):
        groups[f"unmarked_{sub}"] = [i for i in groups["unmarked_decisive"] if new[i]["substratum"] == sub]
    arms = ("DET", "AF-M", "AF-MS", "FETCH-ALL", "HY", "B3", "AO", "B6")
    table = {}
    for g, ids in groups.items():
        table[g] = {"n": len(ids)}
        for arm in arms:
            k = sum(nmaj(arm, i) for i in ids)
            tl = [ntools[arm][i] if isinstance(ntools[arm].get(i), int) else (sum(ntools[arm][i]) / len(ntools[arm][i]) if ntools[arm].get(i) else 0) for i in ids]
            table[g][arm] = {"correct": k, "wilson": wilson(k, len(ids)), "tool_calls_per_case": round(sum(tl) / max(1, len(tl)), 2)}
    res["new_cases"] = table
    res["HY_trigger_rate"] = {g: f"{sum(any(ninv.get(i, [])) for i in ids)}/{len(ids)}" for g, ids in groups.items()}
    res["HY_stability_new"] = sum(len(set(nacts['HY'][i])) == 1 for i in n_ids if nacts['HY'].get(i))
    res["llm_errors"] = dict(nerr)

    # H3
    U = groups["unmarked_decisive"]
    M = {arm: {i: nmaj(arm, i) for i in n_ids} for arm in arms}
    t1, t2 = mcnemar(M["HY"], M["AF-MS"], U), mcnemar(M["HY"], M["AF-M"], U)
    adj = holm([t1["p_exact"], t2["p_exact"]])
    res["H3"] = {"HY_vs_AF-MS": {**t1, "p_holm": adj[0]}, "HY_vs_AF-M": {**t2, "p_holm": adj[1]},
                 "HY_vs_FETCH-ALL": mcnemar(M["HY"], M["FETCH-ALL"], U), "B6_vs_AF-MS": mcnemar(M["B6"], M["AF-MS"], U),
                 "AO_vs_AF-MS": mcnemar(M["AO"], M["AF-MS"], U)}

    # H4
    S = groups["standard"]
    res["H4_new_standard"] = {"HY_vs_DET": mcnemar(M["HY"], M["DET"], S), "HY_vs_B6": mcnemar(M["HY"], M["B6"], S), "HY_vs_AO": mcnemar(M["HY"], M["AO"], S)}
    pooled = {}
    for arm in ("DET", "HY", "B6", "AO"):
        pooled[arm] = {**{i: E[arm][i] for i in o_ids}, **{i: M[arm][i] for i in S}}
    P = o_ids + S
    res["H4_pooled_standard"] = {"n": len(P), **{arm: sum(pooled[arm].values()) for arm in pooled},
                                 "HY_vs_DET": mcnemar(pooled["HY"], pooled["DET"], P), "HY_vs_B6": mcnemar(pooled["HY"], pooled["B6"], P),
                                 "HY_vs_AO": mcnemar(pooled["HY"], pooled["AO"], P)}
    budget = newdir / "budget.json"
    res["budget"] = json.loads(budget.read_text()) if budget.exists() else None
    res["per_case_new"] = {i: {arm: nacts[arm].get(i) for arm in arms} | {"expert": expert.get(i), "author": author[i]} for i in new}
    (HERE / "analysis.json").write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps({k: v for k, v in res.items() if k not in ("per_case_new",)}, indent=1, default=str)[:12000])


if __name__ == "__main__":
    main()
