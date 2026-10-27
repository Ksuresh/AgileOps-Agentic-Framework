"""Study 10b: three-model agreement (GPT, Qwen3-8B, Llama 3.1 8B) and Study 10-style tests for GPT vs Llama.
Runs model_replication/analyze.py logic on the Llama folder and adds all-three agreement per arm."""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model_replication import analyze as A  # noqa: E402
from external_validation.expert_study.analyze import holm, mcnemar, wilson  # noqa: E402


def load(qdir: Path, cases: dict) -> dict:
    out = defaultdict(lambda: defaultdict(list))
    for f in sorted(qdir.glob("shard_*.jsonl")):
        for line in f.read_text().splitlines():
            r = json.loads(line)
            if r.get("error"):
                continue
            out[r["arm"]][r["case"]].append(r["action"])
            if r["arm"] == "B5":
                out["B3"][r["case"]].append(A.b3(cases[r["case"]], r.get("tool_events")))
    return out


def main() -> None:
    import csv, yaml
    cases = {c["id"]: c for f in ("cases_v1.yaml", "cases_replication_v1.yaml") for c in yaml.safe_load((ROOT / "agentic_experiments" / f).read_text())["cases"]}
    cases.update({c["id"]: c for c in yaml.safe_load((A.ES / "cases_new_v1.yaml").read_text())["cases"]})
    gpt = defaultdict(lambda: defaultdict(list))
    for r in csv.DictReader(A.S6_RUNS.open()):
        if r["action"] and r["arm"] in ("HY", "AO", "B6", "B3"):
            gpt[r["arm"]][r["case"]].append(r["action"])
    for arm in ("HY", "AO", "B6"):
        for line in (A.S7_DIR / f"{arm}.jsonl").read_text().splitlines():
            r = json.loads(line)
            if not r.get("error"):
                gpt[arm][r["case"]].append(r["action"])
                if arm == "AO":
                    gpt["B3"][r["case"]].append(A.b3(cases[r["case"]], r.get("tool_events")))
    qwen = load(ROOT / "model_replication/results/qwen3_8b/main", cases)
    llama = load(ROOT / "model_replication/results/llama3.1_8b/main", cases)
    ids = sorted(cases)
    res = {"n": len(ids)}
    same3, sameGL = {}, {}
    for arm, g in (("AAF", "HY"), ("B3", "B3"), ("B5", "AO"), ("B6", "B6")):
        m = {k: {i: A.majority(src[a][i]) for i in ids} for k, (src, a) in {"gpt": (gpt, g), "qwen": (qwen, arm), "llama": (llama, arm)}.items()}
        same3[arm] = {i: m["gpt"][i] is not None and m["gpt"][i] == m["qwen"][i] == m["llama"][i] for i in ids}
        sameGL[arm] = {i: m["gpt"][i] is not None and m["gpt"][i] == m["llama"][i] for i in ids}
        res[arm] = {"same_all_three": sum(same3[arm].values()), "same_gpt_llama": sum(sameGL[arm].values()),
                    "llama_runs": dict(Counter(len(v) for v in llama[arm].values())),
                    "llama_identical_all_runs": sum(1 for i in ids if llama[arm][i] and len(set(llama[arm][i])) == 1)}
    t = {f"AAF_vs_{b}": mcnemar(same3["AAF"], same3[b], ids) for b in ("B5", "B6")}
    for (k, v), p in zip(t.items(), holm([v["p_exact"] for v in t.values()])):
        v["p_holm"] = p
    res["tests_same_all_three"] = t
    tg = {f"AAF_vs_{b}": mcnemar(sameGL["AAF"], sameGL[b], ids) for b in ("B5", "B6")}
    for (k, v), p in zip(tg.items(), holm([v["p_exact"] for v in tg.values()])):
        v["p_holm"] = p
    res["tests_gpt_vs_llama"] = tg
    (ROOT / "model_replication" / "analysis_three_models.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
