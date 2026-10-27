"""Study 8 analysis exactly as pre-specified in PROTOCOL.md. Usage: python analyze_fx.py [results dir]"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from external_validation.expert_study.analyze import det_arm, holm, mcnemar, wilson  # noqa: E402
from agentic_experiments.hybrid_v2 import run_hybrid_fetch  # noqa: E402


def main() -> None:
    rd = Path(sys.argv[1]) if len(sys.argv) > 1 else sorted((HERE / "results").glob("2026*"))[-1]
    fresh = {c["id"]: c for c in yaml.safe_load((HERE / "cases_fx_v1.yaml").read_text())["cases"]}
    key = yaml.safe_load((HERE / "author_labels.yaml").read_text())
    dev = {c["id"]: c for f in ("agentic_experiments/cases_v1.yaml", "agentic_experiments/cases_replication_v1.yaml", "external_validation/expert_study/cases_new_v1.yaml")
           for c in yaml.safe_load((ROOT / f).read_text())["cases"]}
    dkey = {**{k: c["admissible_actions"] for k, c in dev.items() if "admissible_actions" in c},
            **yaml.safe_load((ROOT / "external_validation/expert_study/author_labels.yaml").read_text())}
    acts = {"fresh": defaultdict(lambda: defaultdict(list)), "dev": defaultdict(lambda: defaultdict(list))}
    tools = {"fresh": defaultdict(lambda: defaultdict(list)), "dev": defaultdict(lambda: defaultdict(list))}
    usage = defaultdict(lambda: {"calls": 0, "tokens": 0, "case_runs": 0, "invoked": 0, "schema_fail": 0, "ref_fail": 0, "overrides": 0})
    errors = Counter()
    for f in sorted(rd.glob("*.jsonl")):
        for line in f.read_text().splitlines():
            r = json.loads(line)
            s, a = r["set"], r["arm"]
            if r.get("error"):
                errors[f"{s}:{a}"] += 1
                continue
            acts[s][a][r["case"]].append(r["action"])
            tools[s][a][r["case"]].append(len(r.get("tool_events") or []))
            u = usage[f"{s}:{a}"]
            u["calls"] += int(r.get("n_calls") or 0); u["tokens"] += int((r.get("usage") or {}).get("total_tokens", 0)); u["case_runs"] += 1
            u["invoked"] += bool(r.get("agentic_invoked")); u["schema_fail"] += r.get("all_schema_valid") is False
            u["ref_fail"] += r.get("all_evidence_refs_valid") is False; u["overrides"] += bool(r.get("governance_override"))
    for s, cases in (("fresh", fresh), ("dev", dev)):
        for cid, c in cases.items():
            arms = ("DET", "AF-M", "AF-MS", "FETCH-ALL") if s == "fresh" else ("DET", "FETCH-ALL")
            for arm in arms:
                a, k = det_arm(c, arm); acts[s][arm][cid] = [a]; tools[s][arm][cid] = [k]
            r = run_hybrid_fetch(c); acts[s]["HY2-F"][cid] = [r["selected_action"]]; tools[s]["HY2-F"][cid] = [len(r["tool_events"])]

    def maj(s, arm, cid, k):
        a = acts[s][arm].get(cid) or []
        return bool(a) and sum(x in k[cid] for x in a) * 2 > len(a)
    CON = [i for i in fresh if fresh[i]["substratum"] == "control"]
    U = [i for i in fresh if fresh[i]["stratum"] == "UNMARKED" and i not in CON]
    S = [i for i in fresh if fresh[i]["stratum"] != "UNMARKED"]
    groups = {"unflagged_decisive": U, "controls": CON, "standard": S}
    for sub in ("absent", "cross", "stale"):
        groups["unflagged_" + sub] = [i for i in U if fresh[i]["substratum"] == sub]
    ARMS = ("DET", "AF-M", "AF-MS", "FETCH-ALL", "HY", "HY2-F", "HY2-L", "B6")
    M = {a: {i: maj("fresh", a, i, key) for i in fresh} for a in ARMS}
    res = {"results_dir": rd.name, "errors": dict(errors), "groups": {}}
    for g, ids in groups.items():
        res["groups"][g] = {"n": len(ids)}
        for a in ARMS:
            k = sum(M[a][i] for i in ids)
            tl = [sum(tools["fresh"][a][i]) / len(tools["fresh"][a][i]) for i in ids if tools["fresh"][a].get(i)]
            res["groups"][g][a] = {"correct": k, "wilson": wilson(k, len(ids)), "tool_calls_per_case": round(sum(tl) / max(1, len(tl)), 2)}
    p1, p2 = mcnemar(M["HY2-L"], M["HY"], U), mcnemar(M["HY2-F"], M["HY"], U)
    adj = holm([p1["p_exact"], p2["p_exact"]])
    res["primary"] = {"HY2-L_vs_HY": {**p1, "p_holm": adj[0]}, "HY2-F_vs_HY": {**p2, "p_holm": adj[1]}}
    res["secondary"] = {"HY2-L_vs_FETCH-ALL": mcnemar(M["HY2-L"], M["FETCH-ALL"], U), "HY2-L_vs_B6": mcnemar(M["HY2-L"], M["B6"], U),
                        "HY2-F_vs_FETCH-ALL": mcnemar(M["HY2-F"], M["FETCH-ALL"], U), "HY2-L_vs_HY2-F": mcnemar(M["HY2-L"], M["HY2-F"], U),
                        "HY2-L_vs_AF-MS": mcnemar(M["HY2-L"], M["AF-MS"], U)}
    R = S + CON
    res["no_regression"] = {"n": len(R), **{a: sum(M[a][i] for i in R) for a in ("HY", "HY2-F", "HY2-L", "DET", "B6")},
                            "HY2-L_vs_HY": mcnemar(M["HY2-L"], M["HY"], R), "HY2-F_vs_HY": mcnemar(M["HY2-F"], M["HY"], R)}
    res["stability"] = {a: f"{sum(len(set(acts['fresh'][a][i])) == 1 for i in fresh if acts['fresh'][a].get(i))}/{len(fresh)}" for a in ("HY", "HY2-L")}
    res["usage"] = {k: {**v, "calls_per_case_run": round(v["calls"] / max(1, v["case_runs"]), 2), "tokens_per_case_run": round(v["tokens"] / max(1, v["case_runs"]))}
                    for k, v in usage.items()}
    res["trigger_rate_fresh"] = {g: f"{sum(1 for i in ids if any(True for _ in [0]) and run_hybrid_fetch(fresh[i])['investigation_invoked'])}/{len(ids)}" for g, ids in groups.items()}
    # development (non-confirmatory): 96 earlier cases on their author keys
    dgroups = {"original48": [i for i in dev if not i.startswith("NV-")],
               "study7_standard": [i for i in dev if i.startswith("NV-") and dev[i]["stratum"] != "UNMARKED"],
               "study7_unflagged_decisive": [i for i in dev if i.startswith("NV-") and dev[i]["stratum"] == "UNMARKED" and i not in {"NV-U19", "NV-U20", "NV-U21", "NV-U24"}]}
    res["development"] = {g: {"n": len(ids), **{a: sum(maj("dev", a, i, dkey) for i in ids) for a in ("DET", "FETCH-ALL", "HY2-F", "HY2-L")}} for g, ids in dgroups.items()}
    bud = rd / "budget.json"
    res["budget"] = json.loads(bud.read_text()) if bud.exists() else None
    res["per_case"] = {i: {a: acts["fresh"][a].get(i) for a in ARMS} | {"key": key[i], "substratum": fresh[i]["substratum"]} for i in fresh}
    (HERE / "analysis_fx.json").write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps({k: v for k, v in res.items() if k != "per_case"}, indent=1, default=str)[:9000])


if __name__ == "__main__":
    main()
