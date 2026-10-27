"""Frontier-model deciders on the outcome studies (PROTOCOL_GPT.md, outcome_study/). Reads only the committed decider
inputs (<study>/decisions/variants.json, built from replicate-1 snapshots); never reads outcomes. One run per arm and
variant with GPT-5.6 Luna through the same bounded runtime and prompts as Studies 6-8:
  B5-GPT      run_agentic_only, reasoning none (the LLM decides)
  B6-GPT      run_agentic_only, reasoning medium, coordinator given AAF's written governance policy (COORD_B6)
  AAF-14-GPT  rule set v1.4 + selective investigation with GPT as the investigator (the rules decide)
For B5-GPT the v1.0 and v1.4 rules also re-decide on its retrieved evidence (B3-GPT, B3-14-GPT, no model calls).
Usage: STUDY=outcome_study14 python outcome_study/decide_gpt.py. Resumable: completed rows are not rerun."""
from __future__ import annotations

import copy
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agentic_experiments import agentic_runtime as AR  # noqa: E402
from agentic_experiments.hybrid import deterministic_assessment, run_agentic_only  # noqa: E402
from agentic_experiments.rules_v14 import deterministic_assessment_v14, run_hybrid_v14  # noqa: E402
from revision_2026 import run_llm_arms as RA  # noqa: E402

STUDY = os.environ["STUDY"]
HERE = ROOT / STUDY
OUT = HERE / "decisions_gpt"
ARMS = os.environ.get("ARMS", "AAF-14-GPT B5-GPT B6-GPT").split()


def b3(v: dict, res: dict, decide) -> str:
    ev = copy.deepcopy(v["evidence"])
    for t in res.get("tool_events") or []:
        if t.get("tool") == v.get("expected_tool") and isinstance(t.get("result"), dict):
            ev = AR.merge_tool_result(ev, t["result"])
    return str(decide(ev)["selected_action"])


def main() -> None:
    variants = json.loads((HERE / "decisions" / "variants.json").read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "gpt_rows.jsonl"
    done = set()
    if path.exists():
        done = {(r["decider"], r["case"]) for r in map(json.loads, path.read_text().splitlines()) if r.get("error") is None}
    probe = RA.Runtime("none")
    probe._call("Reply with a JSON object {\"ok\": true}.", {"ping": 1})
    rt = {"AAF-14-GPT": RA.Runtime("none"), "B5-GPT": RA.Runtime("none"), "B6-GPT": RA.Runtime("medium", RA.COORD_B6)}
    fn = {"AAF-14-GPT": lambda c: run_hybrid_v14(c, rt["AAF-14-GPT"]),
          "B5-GPT": lambda c: run_agentic_only(c, rt["B5-GPT"]), "B6-GPT": lambda c: run_agentic_only(c, rt["B6-GPT"])}

    def one(arm: str, v: dict) -> dict:
        try:
            if RA.BUDGET.stopped:
                raise RuntimeError(f"budget stop: {RA.BUDGET.stopped}")
            res = fn[arm](v)
            row = {"decider": arm, "case": v["id"], "variant": v["variant"], "run": 1, "action": res["selected_action"], **RA.slim(res),
                   "trigger": (res.get("trigger") or {}).get("reasons"), "model": RA.MODEL, "error": None}
            if arm == "B5-GPT":
                row["b3_action"] = b3(v, res, deterministic_assessment)
                row["b3_14_action"] = b3(v, res, deterministic_assessment_v14)
        except Exception as exc:
            row = {"decider": arm, "case": v["id"], "variant": v["variant"], "run": 1, "action": None, "model": RA.MODEL,
                   "error": f"{type(exc).__name__}: {str(exc)[:300]}"}
        print(arm, v["id"], row["action"], "calls", RA.BUDGET.calls, "tokens", RA.BUDGET.tokens, flush=True)
        return row

    with path.open("a") as fh:
        for arm in ARMS:
            todo = [v for v in variants if (arm, v["id"]) not in done]
            # AAF-14 temporarily swaps the module's rule function, so its cases run one at a time
            workers = 1 if arm == "AAF-14-GPT" else int(os.environ.get("WORKERS", "6"))
            with ThreadPoolExecutor(workers) as ex:
                for row in ex.map(lambda v: one(arm, v), todo):
                    fh.write(json.dumps(row, default=str) + "\n"); fh.flush()
    (OUT / f"budget_{os.environ.get('GITHUB_RUN_ID', 'local')}.json").write_text(json.dumps({"calls": RA.BUDGET.calls, "tokens": RA.BUDGET.tokens, "stopped": RA.BUDGET.stopped,
                                                 "caps": [RA.MAX_CALLS, RA.MAX_TOKENS], "model": RA.MODEL}, indent=1))


if __name__ == "__main__":
    main()
