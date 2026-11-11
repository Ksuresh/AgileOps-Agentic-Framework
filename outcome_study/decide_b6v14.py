"""LLM decider given AAF's full v1.4 policy (PROTOCOL_B6V14.md). Reads only the committed decider inputs
(<study>/decisions/variants.json, built from replicate-1 snapshots); never reads outcomes.

Two arms, each run with GPT-5.6 Luna (reasoning medium, one run per variant) and Qwen3-8B (thinking on, three runs):
  B6-14   as B6 (LLM domain agents report to an LLM coordinator whose choice is final), but the coordinator is given
          the complete rule set v1.4: the v1.0 policy of B6 plus the release-regression rule R1 and the
          security-over-cost rule R2, in v1.4's priority order. Only the policy differs from B6.
  B6-14E  as B6-14, and the coordinator also receives the raw evidence packet that AAF's rules read (after any tool
          result the agents retrieved), so that it is not limited to what the agents chose to report.
Decider labels: B6-14, B6-14E (Qwen3-8B); B6-14-GPT, B6-14E-GPT (GPT-5.6 Luna).
Uses the unchanged bounded runtime, agent prompts and coordinator prompt of Studies 6-16; only the appended policy text and,
for B6-14E, the extra coordinator input differ. Resumable: completed rows are not rerun.
Usage: STUDY=outcome_study14 KIND=gpt|qwen [ARMS="B6-14 B6-14E"] [SHARD=0 NSHARDS=1] python outcome_study/decide_b6v14.py"""
from __future__ import annotations

import json
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agentic_experiments import agentic_runtime as AR  # noqa: E402
from agentic_experiments import hybrid as H  # noqa: E402
from revision_2026 import run_llm_arms as RA  # noqa: E402

GOVERNANCE_RULES_V14 = RA.GOVERNANCE_RULES + """4. Rule set v1.4 revisions. Apply them after steps 1-3; they take precedence over the action chosen there, R1 before R2.
   R1 (release regression): if the deployment changed recently (recent_release is true) and either
      - the release made reliability worse than before it: p95 latency >= 450 ms and at least twice the pre-release p95
        latency, or error rate >= 8% and at least 5 percentage points above the pre-release error rate; or
      - the new pods restart repeatedly (restart_loops >= 3),
      -> "Rollback to stable deployment". If no pre-release values are available, treat material reliability after a recent
      release as a regression. A problem that was already present before the release does not trigger R1.
   R2 (security over cost): otherwise, if security is material and the action from steps 1-3 is "Scale adjustment",
      "Review scaling policy" or "No action (observe)" -> "Patch or block release".
"""
COORD_B6_V14 = AR.COORDINATOR_SYSTEM + GOVERNANCE_RULES_V14

STUDY = os.environ["STUDY"]
KIND = os.environ.get("KIND", "gpt")
HERE = ROOT / STUDY
OUT = HERE / "decisions_b6v14"
ARMS = os.environ.get("ARMS", "B6-14 B6-14E").split()
SUFFIX = "-GPT" if KIND == "gpt" else ""


def make_runtime(with_evidence: bool):
    if KIND == "gpt":
        base, args = RA.Runtime, ("medium", COORD_B6_V14)
    else:
        from model_replication.run_qwen import QwenRuntime
        base, args = QwenRuntime, (True, COORD_B6_V14)

    class Runtime(base):
        coord_evidence = None

        def _call(self, instructions, payload):
            if with_evidence and instructions == AR.COORDINATOR_SYSTEM and self.coord_evidence is not None:
                payload = dict(payload, incident_evidence=self.coord_evidence)
            return super()._call(instructions, payload)
    return Runtime(*args)


def run_b6(case: dict, runtime, with_evidence: bool) -> dict:
    """hybrid.run_agentic_only, with the coordinator optionally given the evidence packet after tool merges."""
    import copy
    evidence = copy.deepcopy(case["evidence"])
    available = list(case.get("available_tools", []) or [])
    calls, outputs, tool_events = [], [], []
    for agent in ["DevOps", "SRE", "FinOps", "DevSecOps"]:
        call = runtime.domain_agent(agent, evidence, available)
        calls.append(call)
        requested = list(call.output.get("requested_tools", []) or [])
        result, tool = H._execute_requested_tool(case, requested)
        if result is not None and tool is not None:
            tool_events.append({"agent": agent, "tool": tool, "result": result})
            if tool == case.get("expected_tool"):
                evidence = H.merge_tool_result(evidence, result)
                second = runtime.domain_agent(agent, evidence, available, [result])
                calls.append(second)
                call = second
        outputs.append(call.output)
    known = H.packet_ids(H.evidence_packet(evidence)) | {str(x["result"].get("id")) for x in tool_events
                                                         if isinstance(x.get("result"), dict) and x["result"].get("id")}
    if with_evidence:
        runtime.coord_evidence = H.evidence_packet(evidence)
    coord = runtime.coordinator(outputs, known)
    calls.append(coord)
    return {"selected_action": str(coord.output.get("selected_action", "Escalate for evidence/human review")),
            "agent_outputs": outputs, "coordinator": coord.output, "tool_events": tool_events,
            "usage": H.total_call_usage(calls), "latency_ms": sum(x.latency_ms for x in calls),
            "all_schema_valid": all(x.schema_valid for x in calls),
            "all_evidence_refs_valid": all(x.valid_evidence_refs for x in calls),
            "call_audit": [x.audit_record() for x in calls]}


def main() -> None:
    variants = json.loads((HERE / "decisions" / "variants.json").read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    runs = (1,) if KIND == "gpt" else (1, 2, 3)
    shard, nshards = int(os.environ.get("SHARD", "0")), int(os.environ.get("NSHARDS", "1"))
    path = OUT / (f"gpt_rows.jsonl" if KIND == "gpt" else f"qwen_shard_{shard:02d}.jsonl")
    done = set()
    if path.exists():
        done = {(r["decider"], r["case"], r["run"]) for r in map(json.loads, path.read_text().splitlines()) if r.get("error") is None}
    jobs = [(arm, v, run) for v in variants for arm in ARMS for run in runs]
    random.Random(9).shuffle(jobs)
    jobs = [j for j in jobs[shard::nshards] if (j[0] + SUFFIX, j[1]["id"], j[2]) not in done]
    print(f"{STUDY} {KIND}: {len(jobs)} jobs to run", flush=True)
    if KIND == "gpt":
        RA.Runtime("none")._call("Reply with a JSON object {\"ok\": true}.", {"ping": 1})

    def one(job):
        arm, v, run = job
        name = arm + SUFFIX
        try:
            if KIND == "gpt" and RA.BUDGET.stopped:
                raise RuntimeError(f"budget stop: {RA.BUDGET.stopped}")
            res = run_b6(v, make_runtime(arm == "B6-14E"), arm == "B6-14E")
            return {"decider": name, "case": v["id"], "variant": v["variant"], "run": run, "action": res["selected_action"],
                    **RA.slim(res), "model": RA.MODEL if KIND == "gpt" else os.environ.get("QWEN_MODEL", "qwen3:8b"), "error": None}
        except Exception as exc:
            return {"decider": name, "case": v["id"], "variant": v["variant"], "run": run, "action": None,
                    "error": f"{type(exc).__name__}: {str(exc)[:300]}"}

    with path.open("a") as fh:
        if KIND == "gpt":
            with ThreadPoolExecutor(int(os.environ.get("WORKERS", "6"))) as ex:
                for row in ex.map(one, jobs):
                    fh.write(json.dumps(row, default=str) + "\n"); fh.flush()
                    print(row["decider"], row["case"], row["run"], row["action"] or row["error"], flush=True)
        else:
            for job in jobs:
                row = one(job)
                fh.write(json.dumps(row, default=str) + "\n"); fh.flush()
                print(row["decider"], row["case"], row["run"], row["action"] or row["error"], flush=True)
    if KIND == "gpt":
        print("budget:", RA.BUDGET.calls, "calls", RA.BUDGET.tokens, "tokens", RA.BUDGET.stopped or "", flush=True)


if __name__ == "__main__":
    main()
