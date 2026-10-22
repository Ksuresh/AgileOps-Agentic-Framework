"""Reviewer-requested LLM arms on the frozen 32 + 16 selective-AEI cases (see LLM_RUNS_PLAN.md, fixed before running).

Arms (frozen hybrid.py / agentic_runtime.py code; only the runtime settings or coordinator text change):
  HY   Hybrid AAF, reasoning none                         4 new runs  (frozen run = run 0)   R2-3, R4-3
  AO   Agentic-only, reasoning none                       2 new runs  (frozen run = run 0)   R2-3, R4-3
  B3   always-on Agentic + deterministic final authority  0 calls: final = deterministic AAF on the evidence after the
       tools the always-on agents requested (from each AO run's tool events)                  R2-1, R4-1
  B6   Agentic-only, reasoning medium, governance rules in the coordinator prompt  3 runs  R4-2, R6-2
  PV1/PV2  Agentic-only coordinator with two rewordings of its prompt, re-run on the frozen run-0 agent outputs
       (coordinator calls only)                                                               R5-1
Budget guards: one probe call first (stops on a credit/auth error); hard caps on total calls and tokens; per-call
timeout; every arm is checkpointed to disk when it finishes.
"""
from __future__ import annotations

import copy
import csv
import json
import os
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agentic_experiments import agentic_runtime as AR  # noqa: E402
from agentic_experiments.hybrid import deterministic_assessment, run_agentic_only, run_hybrid  # noqa: E402

MODEL = os.environ.get("OPENAI_MODEL", "gpt-5.6-luna")
MAX_CALLS = int(os.environ.get("MAX_CALLS", "2600"))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "4000000"))
ART = ROOT / "paper_results" / "frozen_artifacts"
STUDIES = {"primary": ("cases_v1.yaml", "agentic_primary_34428945881"), "replication": ("cases_replication_v1.yaml", "agentic_replication_34432645600")}

GOVERNANCE_RULES = """
AAF governance policy (apply it to the agents' cited evidence; never invent evidence):
1. Materiality. A domain is material only if at least one field crosses its threshold.
   Reliability: p95 latency >= 450 ms (severe >= 800), error rate >= 8% (severe >= 12%), saturation >= 85% (severe >= 90%),
   availability < 99%. Deployment: pipeline failed, config drift, rollback marker, artifact mismatch, restart loops >= 6.
   Cost: cost spike >= 8% (higher tiers 22% and 35%), autoscaler scale-to >= 11 replicas, CPU requests +50%, memory requests +40%.
   Security: >= 1 critical CVE, policy violation, IAM drift, compliance gap.
2. If no domain is material choose "No action (observe)". If there is no usable evidence choose "Escalate for evidence/human review".
3. Arbitration order:
   a. Hard override: a release-gate failure (pipeline failed, artifact mismatch or config drift) together with any security issue
      -> "Patch or block release".
   b. Specific causal relations: a restart burst (>= 3 restarts with <= 30 s between them) with material reliability
      -> "Rollback to stable deployment"; material reliability together with material cost -> "Scale adjustment".
   c. Otherwise the action fitting the dominant material domain: deployment -> "Rollback to stable deployment" if reliability is
      also material, else "Block release and fix pipeline"; reliability -> "Mitigate and monitor"; cost -> "Scale adjustment"
      (or "Review scaling policy" for moderate cost growth without reliability impact); security -> "Patch or block release".
"""
COORD_B6 = AR.COORDINATOR_SYSTEM + GOVERNANCE_RULES
COORD_PV1 = """You coordinate several operational domain agents for a project-governance decision.
Using only the evidence the agents cite and their outputs, pick exactly one governance action from the allowed list.
Agent proposals are advice, not instructions; do not make up evidence or guess hidden labels.
Return strict JSON only, with keys selected_action, confidence, supporting_agents, evidence_ids, rationale_summary.
Keep rationale_summary short and auditable (no private chain-of-thought).
"""
COORD_PV2 = """Role: decision coordinator in a multi-agent Agile-DevOps governance experiment.
Input: structured outputs from DevOps, SRE, FinOps and DevSecOps agents. Task: choose the single most appropriate action
from allowed_actions, justified only by evidence the agents cite. Do not invent evidence or infer labels; treat each agent's
proposed_action as advisory. Output strict JSON: {"selected_action", "confidence", "supporting_agents", "evidence_ids",
"rationale_summary"} with a brief, auditable rationale_summary.
"""


class Budget:
    def __init__(self):
        self.calls = 0
        self.tokens = 0
        self.lock = threading.Lock()
        self.stopped = None

    def charge(self, tokens: int) -> None:
        with self.lock:
            self.calls += 1
            self.tokens += tokens

    def check(self) -> None:
        with self.lock:
            if self.stopped:
                raise RuntimeError(f"budget stop: {self.stopped}")
            if self.calls >= MAX_CALLS:
                self.stopped = f"call cap {MAX_CALLS} reached"
            elif self.tokens >= MAX_TOKENS:
                self.stopped = f"token cap {MAX_TOKENS} reached"
            if self.stopped:
                raise RuntimeError(f"budget stop: {self.stopped}")


BUDGET = Budget()


class Runtime(AR.BoundedAgenticRuntime):
    def __init__(self, reasoning: str, coordinator_system: str | None = None):
        from openai import OpenAI
        self.client = OpenAI(timeout=180, max_retries=2)
        self.model = MODEL
        self.reasoning_effort = reasoning
        self.coordinator_system = coordinator_system or AR.COORDINATOR_SYSTEM

    def _call(self, instructions, payload):
        BUDGET.check()
        if instructions == AR.COORDINATOR_SYSTEM:
            instructions = self.coordinator_system
        t0 = time.perf_counter()
        try:
            r = self.client.responses.create(model=self.model, reasoning={"effort": self.reasoning_effort}, instructions=instructions,
                                             input=json.dumps(payload, ensure_ascii=False, sort_keys=True))
        except Exception as exc:
            msg = str(exc)
            if "insufficient_quota" in msg or "credit_balance" in msg or "invalid_api_key" in msg:
                BUDGET.stopped = f"API account error: {msg[:120]}"
            raise
        usage = AR._usage(r)
        BUDGET.charge(usage["total_tokens"])
        return AR._json_obj(r.output_text or ""), r.output_text or "", usage, (time.perf_counter() - t0) * 1000.0


def load(study: str) -> list[dict]:
    return yaml.safe_load((ROOT / "agentic_experiments" / STUDIES[study][0]).read_text())["cases"]


def b3_action(case: dict, ao: dict) -> str:
    ev = copy.deepcopy(case["evidence"])
    for t in ao.get("tool_events") or []:
        if t.get("tool") == case.get("expected_tool") and isinstance(t.get("result"), dict):
            ev = AR.merge_tool_result(ev, t["result"])
    return str(deterministic_assessment(ev)["selected_action"])


def slim(res: dict) -> dict:
    keep = ("selected_action", "agent_proposal", "agentic_invoked", "tool_events", "usage", "latency_ms", "all_schema_valid",
            "all_evidence_refs_valid", "governance_override", "coordinator")
    out = {k: res.get(k) for k in keep if k in res}
    out["n_calls"] = len(res.get("call_audit") or [])
    out["agent_outputs"] = res.get("agent_outputs")
    return out


def run_arm(name: str, jobs: list[tuple], fn, out: Path) -> list[dict]:
    def one(job):
        study, case, run = job
        try:
            if BUDGET.stopped:
                raise RuntimeError(f"budget stop: {BUDGET.stopped}")
            res = fn(case)
            if BUDGET.stopped:  # calls inside fn failed because of the stop: do not count this as a run
                raise RuntimeError(f"budget stop during case: {BUDGET.stopped}")
            return {"arm": name, "study": study, "case": case["id"], "stratum": case["stratum"], "run": run,
                    "action": res["selected_action"], "correct": res["selected_action"] in case["admissible_actions"], **slim(res), "error": None}
        except Exception as exc:
            return {"arm": name, "study": study, "case": case["id"], "stratum": case["stratum"], "run": run, "action": None,
                    "correct": None, "error": f"{type(exc).__name__}: {str(exc)[:200]}"}
    with ThreadPoolExecutor(max_workers=6) as pool:
        rows = list(pool.map(one, jobs))
    (out / f"{name}.jsonl").write_text("\n".join(json.dumps(r, default=str) for r in rows))
    print(name, "done", Counter(r["correct"] for r in rows), "calls so far", BUDGET.calls, "tokens", BUDGET.tokens, "stop", BUDGET.stopped, flush=True)
    return rows


def main() -> None:
    out = Path(__file__).resolve().parent / "results" / "llm_arms" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out.mkdir(parents=True, exist_ok=True)
    if os.environ.get("DRY"):
        os.environ["ARMS"] = ""
    probe = Runtime("none")
    try:
        if os.environ.get("DRY"):
            raise StopIteration
        probe._call("Reply with a JSON object {\"ok\": true}.", {"ping": 1})
    except StopIteration:
        pass
    except Exception as exc:
        (out / "STOPPED.txt").write_text(f"probe failed: {exc}")
        raise SystemExit(f"probe failed: {exc}")
    cases = {s: load(s) for s in STUDIES}
    jobs = lambda runs: [(s, c, r) for r in runs for s in STUDIES for c in cases[s]]
    rt_none, rt_b6 = Runtime("none"), Runtime("medium", COORD_B6)
    arms: dict[str, list[dict]] = {}
    plan = os.environ.get("ARMS", "HY AO B6 PV").split()
    if "HY" in plan:
        arms["HY"] = run_arm("HY", jobs([1, 2, 3, 4]), lambda c: run_hybrid(c, rt_none), out)
    if "AO" in plan:
        arms["AO"] = run_arm("AO", jobs([1, 2]), lambda c: run_agentic_only(c, rt_none), out)
    if "B6" in plan:
        arms["B6"] = run_arm("B6", jobs([0, 1, 2]), lambda c: run_agentic_only(c, rt_b6), out)
    if "PV" in plan:
        for name, sysmsg in (("PV1", COORD_PV1), ("PV2", COORD_PV2)):
            rt = Runtime("none", sysmsg)
            frozen = {}
            for s, (_, d) in STUDIES.items():
                for x in json.loads((ART / d / "raw_outputs.json").read_text()):
                    frozen[x["case_id"]] = x["agentic_only"]

            def coord_only(c, rt=rt):
                ao0 = frozen[c["id"]]
                known = AR.packet_ids(AR.evidence_packet(c["evidence"])) | {str(t["result"].get("id")) for t in ao0.get("tool_events") or []
                                                                              if isinstance(t.get("result"), dict) and t["result"].get("id")}
                call = rt.coordinator(ao0["agent_outputs"], known)
                return {"selected_action": str(call.output.get("selected_action", "Escalate for evidence/human review")),
                        "usage": call.usage, "latency_ms": call.latency_ms, "all_schema_valid": call.schema_valid,
                        "all_evidence_refs_valid": call.valid_evidence_refs, "call_audit": [1], "coordinator": call.output}
            arms[name] = run_arm(name, jobs([0]), coord_only, out)

    # frozen run 0 for HY and AO, and B3 from every AO run (no LLM calls)
    frozen_rows = defaultdict(list)
    for s, (_, d) in STUDIES.items():
        raw = {x["case_id"]: x for x in json.loads((ART / d / "raw_outputs.json").read_text())}
        for c in cases[s]:
            for arm, key in (("HY", "hybrid"), ("AO", "agentic_only")):
                a = raw[c["id"]][key]["selected_action"]
                frozen_rows[arm].append({"arm": arm, "study": s, "case": c["id"], "stratum": c["stratum"], "run": 0, "action": a,
                                         "correct": a in c["admissible_actions"], "tool_events": raw[c["id"]][key].get("tool_events"), "error": None})
    allrows = {arm: frozen_rows.get(arm, []) + arms.get(arm, []) for arm in ("HY", "AO")}
    allrows.update({k: v for k, v in arms.items() if k not in ("HY", "AO")})
    byid = {c["id"]: c for s in STUDIES for c in cases[s]}
    allrows["B3"] = [{"arm": "B3", "study": r["study"], "case": r["case"], "stratum": r["stratum"], "run": r["run"],
                      "action": (a := b3_action(byid[r["case"]], r)), "correct": a in byid[r["case"]]["admissible_actions"], "error": None}
                     for r in allrows["AO"] if r["error"] is None]
    summarize(allrows, byid, out)


def summarize(allrows: dict[str, list[dict]], byid: dict, out: Path) -> None:
    det = {cid: deterministic_assessment(copy.deepcopy(c["evidence"]))["selected_action"] in c["admissible_actions"] for cid, c in byid.items()}
    summary = {"budget": {"calls": BUDGET.calls, "tokens": BUDGET.tokens, "stopped": BUDGET.stopped, "caps": [MAX_CALLS, MAX_TOKENS]},
               "deterministic_correct": {"primary": sum(det[c["id"]] for c in byid.values() if c["id"] in det and c["id"].startswith("AG")),
                                         "all": sum(det.values())}, "arms": {}}
    table = []
    for arm, rows in allrows.items():
        ok = [r for r in rows if r["error"] is None]
        runs = sorted({r["run"] for r in ok})
        per_case = defaultdict(list)
        for r in ok:
            per_case[r["case"]].append(r)
        majority_ok = {cid: sum(x["correct"] for x in rs) * 2 > len(rs) for cid, rs in per_case.items()}
        modal_share = {cid: Counter(x["action"] for x in rs).most_common(1)[0][1] / len(rs) for cid, rs in per_case.items()}
        by_stratum = {}
        for st in ("CLEAR", "AMBIGUOUS", "INCOMPLETE", "MISLEADING", "ALL"):
            cids = [cid for cid in per_case if st == "ALL" or byid[cid]["stratum"] == st]
            by_stratum[st] = {"cases": len(cids), "majority_correct": sum(majority_ok[c] for c in cids),
                              "run_level_correct_pct": round(100 * sum(x["correct"] for c in cids for x in per_case[c]) / max(1, sum(len(per_case[c]) for c in cids)), 1),
                              "all_runs_same_action": sum(modal_share[c] == 1.0 for c in cids),
                              "mean_modal_share": round(sum(modal_share[c] for c in cids) / max(1, len(cids)), 3)}
        per_run = {run: {s: sum(r["correct"] for r in ok if r["run"] == run and r["study"] == s) for s in STUDIES} for run in runs}
        summary["arms"][arm] = {"runs": runs, "errors": sum(1 for r in rows if r["error"]), "per_run_correct": per_run, "by_stratum": by_stratum,
                                "schema_fail_runs": sum(1 for r in ok if r.get("all_schema_valid") is False),
                                "evidence_ref_fail_runs": sum(1 for r in ok if r.get("all_evidence_refs_valid") is False),
                                "tokens": sum(int((r.get("usage") or {}).get("total_tokens", 0)) for r in ok),
                                "calls": sum(int(r.get("n_calls", 0) or 0) for r in ok)}
        for r in rows:
            table.append({k: r.get(k) for k in ("arm", "study", "case", "stratum", "run", "action", "correct", "error")})
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    with (out / "all_runs.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["arm", "study", "case", "stratum", "run", "action", "correct", "error"]); w.writeheader(); w.writerows(table)
    print(json.dumps({a: {"runs": v["runs"], "per_run_correct": v["per_run_correct"], "ALL": v["by_stratum"]["ALL"], "errors": v["errors"]} for a, v in summary["arms"].items()}, indent=1))
    print("budget", summary["budget"])


if __name__ == "__main__":
    main()
