"""Study 10 analysis (PROTOCOL.md): cross-model agreement (H10.1), run stability (H10.2), accuracy (H10.3), containment
(H10.4). GPT actions: frozen Study 6 runs (revision_2026 all_runs.csv) and Study 7 runs (expert_study results).
Qwen actions: model_replication/results/<model>/<label>/shard_*.jsonl."""
from __future__ import annotations

import copy
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agentic_experiments.agentic_runtime import merge_tool_result  # noqa: E402
from agentic_experiments.hybrid import deterministic_assessment  # noqa: E402
from external_validation.expert_study.analyze import holm, mcnemar, wilson  # noqa: E402

ES = ROOT / "external_validation" / "expert_study"
S6_RUNS = ROOT / "revision_2026/results/llm_arms/20261022T064831Z/all_runs.csv"
S7_DIR = sorted((ES / "results").glob("2026*"))[-1]
ARM_GPT = {"AAF": "HY", "B5": "AO", "B6": "B6", "B3": "B3"}


def b3(case, tool_events):
    ev = copy.deepcopy(case["evidence"])
    for t in tool_events or []:
        if t.get("tool") == case.get("expected_tool") and isinstance(t.get("result"), dict):
            ev = merge_tool_result(ev, t["result"])
    return str(deterministic_assessment(ev)["selected_action"])


def majority(actions: list[str]) -> str | None:
    if not actions:
        return None
    c = Counter(actions).most_common()
    top = [a for a, n in c if n == c[0][1]]
    return sorted(top)[0] if len(top) > 1 else top[0]


def main() -> None:
    qdir = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "model_replication/results/qwen3_8b/main"
    cases = {c["id"]: c for f in ("cases_v1.yaml", "cases_replication_v1.yaml") for c in yaml.safe_load((ROOT / "agentic_experiments" / f).read_text())["cases"]}
    new = {c["id"]: c for c in yaml.safe_load((ES / "cases_new_v1.yaml").read_text())["cases"]}
    cases.update(new)
    author = {**{k: c["admissible_actions"] for k, c in cases.items() if "admissible_actions" in c}, **yaml.safe_load((ES / "author_labels.yaml").read_text())}
    cmap = json.loads((ES / "label_map.json").read_text())
    pract = {cmap[k]["case"]: v["acceptable"] for k, v in json.loads((ES / "expert_labels.json").read_text()).items() if not v.get("cannot")}

    gpt = defaultdict(lambda: defaultdict(list))
    for r in csv.DictReader(S6_RUNS.open()):
        if r["action"] and r["arm"] in ("HY", "AO", "B6", "B3"):
            gpt[r["arm"]][r["case"]].append(r["action"])
    for arm in ("HY", "AO", "B6"):
        f = S7_DIR / f"{arm}.jsonl"
        for line in f.read_text().splitlines():
            r = json.loads(line)
            if not r.get("error"):
                gpt[arm][r["case"]].append(r["action"])
                if arm == "AO":
                    gpt["B3"][r["case"]].append(b3(new[r["case"]], r.get("tool_events")))

    qwen = defaultdict(lambda: defaultdict(list))
    errors, malformed, aaf_malformed_changed = Counter(), Counter(), 0
    for f in sorted(qdir.glob("shard_*.jsonl")):
        for line in f.read_text().splitlines():
            r = json.loads(line)
            if r.get("error"):
                errors[r["arm"]] += 1
                continue
            qwen[r["arm"]][r["case"]].append(r["action"])
            if r.get("all_schema_valid") is False:
                malformed[r["arm"]] += 1
                if r["arm"] == "AAF" and r["action"] != str(deterministic_assessment(cases[r["case"]]["evidence"])["selected_action"]) \
                        and not any(t.get("tool") == cases[r["case"]].get("expected_tool") for t in r.get("tool_events") or []):
                    aaf_malformed_changed += 1
            if r["arm"] == "B5":
                qwen["B3"][r["case"]].append(b3(cases[r["case"]], r.get("tool_events")))
    ids = sorted(cases)
    res = {"qwen_dir": str(qdir.relative_to(ROOT)), "errors": dict(errors), "n_cases": len(ids),
           "qwen_runs_per_arm": {a: Counter(len(v) for v in qwen[a].values()) for a in qwen}}

    # H10.1 cross-model agreement of majority actions
    same = {}
    for arm in ("AAF", "B3", "B5", "B6"):
        same[arm] = {i: (majority(qwen[arm][i]) is not None and majority(qwen[arm][i]) == majority(gpt[ARM_GPT[arm] if arm != "B3" else "B3"][i])) for i in ids}
    t = {"AAF_vs_B5": mcnemar(same["AAF"], same["B5"], ids), "AAF_vs_B6": mcnemar(same["AAF"], same["B6"], ids)}
    adj = holm([t["AAF_vs_B5"]["p_exact"], t["AAF_vs_B6"]["p_exact"]])
    t["AAF_vs_B5"]["p_holm"], t["AAF_vs_B6"]["p_holm"] = adj
    res["H10_1_cross_model_same_action"] = {a: {"k": sum(same[a].values()), "n": len(ids), "wilson": wilson(sum(same[a].values()), len(ids))} for a in same}
    res["H10_1_tests"] = t

    # H10.2 stability within Qwen
    res["H10_2_identical_in_all_runs"] = {a: sum(1 for i in ids if qwen[a][i] and len(set(qwen[a][i])) == 1) for a in ("AAF", "B3", "B5", "B6")}

    # H10.3 accuracy (majority) per key
    det = {i: str(deterministic_assessment(cases[i]["evidence"])["selected_action"]) for i in ids}
    acc = {}
    for key_name, key, subset in (("authors_s6", author, [i for i in ids if i not in new]), ("authors_s7", author, [i for i in ids if i in new]),
                                  ("practitioner_s6", pract, [i for i in ids if i not in new and i in pract]),
                                  ("practitioner_s7", pract, [i for i in ids if i in new and i in pract])):
        ok = {"B1": {i: det[i] in key[i] for i in subset}}
        for a in ("AAF", "B3", "B5", "B6"):
            ok[a] = {i: (majority(qwen[a][i]) in key[i]) for i in subset}
            ok[a + "_gpt"] = {i: (majority(gpt[ARM_GPT[a]][i]) in key[i]) for i in subset}
        acc[key_name] = {"n": len(subset), **{a: sum(v.values()) for a, v in ok.items()},
                         "tests_qwen": {f"AAF_vs_{b}": mcnemar(ok["AAF"], ok[b], subset) for b in ("B1", "B5", "B6", "B3")}}
    res["H10_3_accuracy"] = acc
    res["H10_4_malformed_case_runs"] = dict(malformed)
    res["H10_4_aaf_malformed_changed_final_action_without_tool_result"] = aaf_malformed_changed
    out = ROOT / "model_replication" / "analysis.json"
    out.write_text(json.dumps(res, indent=2, default=str))
    print(json.dumps(res, indent=2, default=str)[:6000])


if __name__ == "__main__":
    main()
