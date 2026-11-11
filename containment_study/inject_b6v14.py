"""Study 11c: the Study 11b prompt injection against the LLM decider given the full v1.4 policy (PROTOCOL_11c.md).

Same 96 cases and the same injected note as Study 11b (model_replication/run_qwen.inject), with GPT-5.6 Luna, one run
per arm and case:
  AAF-14-GPT  rule set v1.4 decides, GPT-5.6 Luna investigates (as in the outcome studies)
  B6-GPT      LLM decides, coordinator given AAF's v1.0 policy (as in Studies 6 and 11b)
  B6-14E-GPT  LLM decides, coordinator given the full v1.4 policy and the evidence packet (outcome_study/decide_b6v14.py)
The injected note is an ordinary evidence field, so it reaches every LLM that sees the evidence, including the B6-14E
coordinator's evidence packet; the deterministic rules ignore it. Resumable.
Amendment 1 (PROTOCOL_11c.md, before any Qwen run): KIND=qwen runs the same attack with Qwen3-8B (Ollama, settings of
Study 11b), one run per case, arms AAF-14-Q (rule set v1.4 decides, Qwen investigates, thinking off) and B6-14E-Q (LLM
decides with the full v1.4 policy and the evidence packet, thinking on), in shards (SHARD / NSHARDS).
Usage: python containment_study/inject_b6v14.py   (env: KIND=gpt|qwen, ARMS, WORKERS, MAX_CALLS, MAX_TOKENS, SHARD, NSHARDS)"""
from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("STUDY", "outcome_study14")  # decide_b6v14 reads STUDY at import; not used here
from agentic_experiments.hybrid import run_agentic_only  # noqa: E402
from agentic_experiments.rules_v14 import run_hybrid_v14  # noqa: E402
from model_replication.run_qwen import CASE_FILES, inject  # noqa: E402
from outcome_study import decide_b6v14 as D  # noqa: E402
from revision_2026 import run_llm_arms as RA  # noqa: E402

OUT = ROOT / "containment_study" / "results" / "injection_b6v14"
KIND = os.environ.get("KIND", "gpt")
ARMS = os.environ.get("ARMS", "AAF-14-GPT B6-GPT B6-14E-GPT" if KIND == "gpt" else "AAF-14-Q B6-14E-Q").split()


def cases() -> list[tuple[str, dict]]:
    return [(study, inject(c)) for study, path in CASE_FILES.items() for c in yaml.safe_load(path.read_text())["cases"]]


def main_qwen() -> None:
    import random
    from model_replication.run_qwen import QwenRuntime
    OUT.mkdir(parents=True, exist_ok=True)
    shard, nshards = int(os.environ.get("SHARD", "0")), int(os.environ.get("NSHARDS", "1"))
    path = OUT / f"qwen_shard_{shard:02d}.jsonl"
    done = set()
    if path.exists():
        done = {(r["arm"], r["case"]) for r in map(json.loads, path.read_text().splitlines()) if r.get("error") is None}
    D.KIND = "qwen"
    fn = {"AAF-14-Q": lambda c: run_hybrid_v14(c, QwenRuntime(False)),
          "B6-14E-Q": lambda c: D.run_b6(c, D.make_runtime(True), True)}
    jobs = [(arm, s, c) for s, c in cases() for arm in ARMS]
    random.Random(11).shuffle(jobs)
    jobs = [j for j in jobs[shard::nshards] if (j[0], j[2]["id"]) not in done]
    print(f"qwen injection shard {shard}: {len(jobs)} jobs", flush=True)
    with path.open("a") as fh:
        for arm, study, c in jobs:
            try:
                res = fn[arm](c)
                row = {"arm": arm, "study": study, "case": c["id"], "stratum": c["stratum"], "injected_action": c["_injected_action"],
                       "action": res["selected_action"], **RA.slim(res), "model": os.environ.get("QWEN_MODEL", "qwen3:8b"), "error": None}
            except Exception as exc:
                row = {"arm": arm, "study": study, "case": c["id"], "action": None, "injected_action": c["_injected_action"],
                       "model": os.environ.get("QWEN_MODEL", "qwen3:8b"), "error": f"{type(exc).__name__}: {str(exc)[:300]}"}
            fh.write(json.dumps(row, default=str) + "\n"); fh.flush()
            print(arm, c["id"], row["action"] or row["error"], flush=True)


def main() -> None:
    if KIND == "qwen":
        return main_qwen()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "gpt_rows.jsonl"
    done = set()
    if path.exists():
        done = {(r["arm"], r["case"]) for r in map(json.loads, path.read_text().splitlines()) if r.get("error") is None}
    RA.Runtime("none")._call("Reply with a JSON object {\"ok\": true}.", {"ping": 1})
    D.KIND = "gpt"
    rt_b6 = RA.Runtime("medium", RA.COORD_B6)
    fn = {"AAF-14-GPT": lambda c: run_hybrid_v14(c, RA.Runtime("none")),
          "B6-GPT": lambda c: run_agentic_only(c, rt_b6),
          "B6-14E-GPT": lambda c: D.run_b6(c, D.make_runtime(True), True)}

    def one(job):
        arm, study, c = job
        try:
            if RA.BUDGET.stopped:
                raise RuntimeError(f"budget stop: {RA.BUDGET.stopped}")
            res = fn[arm](c)
            row = {"arm": arm, "study": study, "case": c["id"], "stratum": c["stratum"], "injected_action": c["_injected_action"],
                   "action": res["selected_action"], **RA.slim(res), "model": RA.MODEL, "error": None}
        except Exception as exc:
            row = {"arm": arm, "study": study, "case": c["id"], "action": None, "injected_action": c["_injected_action"],
                   "model": RA.MODEL, "error": f"{type(exc).__name__}: {str(exc)[:300]}"}
        print(arm, c["id"], row["action"] or row["error"], "calls", RA.BUDGET.calls, flush=True)
        return row

    allc = cases()
    with path.open("a") as fh:
        for arm in ARMS:
            todo = [(arm, s, c) for s, c in allc if (arm, c["id"]) not in done]
            workers = 1 if arm == "AAF-14-GPT" else int(os.environ.get("WORKERS", "6"))  # run_hybrid_v14 swaps a module function
            with ThreadPoolExecutor(workers) as ex:
                for row in ex.map(one, todo):
                    fh.write(json.dumps(row, default=str) + "\n"); fh.flush()
    (OUT / f"budget_{os.environ.get('GITHUB_RUN_ID', 'local')}.json").write_text(json.dumps(
        {"calls": RA.BUDGET.calls, "tokens": RA.BUDGET.tokens, "stopped": RA.BUDGET.stopped, "caps": [RA.MAX_CALLS, RA.MAX_TOKENS],
         "model": RA.MODEL}, indent=1))


if __name__ == "__main__":
    main()
