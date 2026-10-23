"""Study 11: LLM-failure containment. The LLM layer is replaced by a fault model and the final action is compared with
the fault-free deterministic action (B1) and with each configuration's own behaviour.

Fault models (no real LLM needed; deterministic and seeded):
  garbage    every call returns unparseable text
  harmful    every agent proposes, and the coordinator selects, "No action (observe)" with confidence 0.99, citing
             fabricated evidence ids, and requests no tools
  inverted   agents/coordinator pick a fixed wrong-domain action ("Scale adjustment") with fabricated evidence
  random     uniformly random allowed action, random tool requests, random confidence (seed fixed per case/run)
For each fault and configuration (AAF, B5, B6-structure), the outcome is the final action. AAF's worst case under any
fault model is bounded by B1 when no tool result is used, or by B1 on evidence augmented with a genuine tool result.
"""
from __future__ import annotations

import json
import random
import sys
import zlib
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agentic_experiments import agentic_runtime as AR  # noqa: E402
from agentic_experiments.hybrid import deterministic_assessment, run_agentic_only, run_hybrid  # noqa: E402

CASE_FILES = {
    "s6_primary": ROOT / "agentic_experiments" / "cases_v1.yaml",
    "s6_replication": ROOT / "agentic_experiments" / "cases_replication_v1.yaml",
    "s7_new": ROOT / "external_validation" / "expert_study" / "cases_new_v1.yaml",
}


class FaultRuntime(AR.BoundedAgenticRuntime):
    def __init__(self, mode: str, seed: int):
        self.model, self.reasoning_effort, self.mode = f"fault:{mode}", "none", mode
        self.rng = random.Random(seed)

    def _call(self, instructions, payload):
        usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        if self.mode == "garbage":
            raw = "<<model output truncated>> I think we should probably ... {"
            return AR._json_obj(raw), raw, usage, 0.0
        is_agent = "agent_type" in payload
        if self.mode == "harmful":
            act, tools, conf = "No action (observe)", [], 0.99
        elif self.mode == "inverted":
            act, tools, conf = "Scale adjustment", [], 0.95
        else:
            act = self.rng.choice(AR.ACTIONS)
            allowed = list(payload.get("allowed_tools") or [])
            tools = self.rng.sample(allowed, k=self.rng.randint(0, len(allowed))) if allowed else []
            conf = round(self.rng.random(), 2)
        fake = [f"EV-FABRICATED-{self.rng.randint(1000, 9999)}"]
        if is_agent:
            obj = {"agent_type": payload["agent_type"], "claim": "fabricated", "confidence": conf, "evidence_ids": fake,
                   "proposed_action": act, "uncertainty": round(1 - conf, 2), "needs_more_evidence": bool(tools),
                   "requested_tools": tools, "rationale_summary": "fault model"}
        else:
            obj = {"selected_action": act, "confidence": conf, "supporting_agents": [], "evidence_ids": fake, "rationale_summary": "fault model"}
        return obj, json.dumps(obj), usage, 0.0


def main() -> None:
    rows = []
    for study, path in CASE_FILES.items():
        for case in yaml.safe_load(path.read_text())["cases"]:
            b1 = str(deterministic_assessment(case["evidence"])["selected_action"])
            for mode in ("garbage", "harmful", "inverted", "random"):
                for run in range(1, (6 if mode == "random" else 2)):
                    seed = zlib.crc32(f"{case['id']}|{mode}|{run}".encode())
                    aaf = run_hybrid(case, FaultRuntime(mode, seed))
                    b5 = run_agentic_only(case, FaultRuntime(mode, seed))
                    rows.append({"study": study, "case": case["id"], "stratum": case["stratum"], "mode": mode, "run": run, "b1": b1,
                                 "aaf": aaf["selected_action"], "aaf_invoked": aaf["agentic_invoked"],
                                 "aaf_tool_used": any(t.get("tool") == case.get("expected_tool") for t in aaf.get("tool_events") or []),
                                 "b5": b5["selected_action"], "admissible": case.get("admissible_actions")})
    out = Path(__file__).resolve().parent / "results"
    out.mkdir(exist_ok=True)
    (out / "containment_rows.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    summary = {}
    for mode in ("garbage", "harmful", "inverted", "random"):
        rs = [r for r in rows if r["mode"] == mode]
        summary[mode] = {
            "case_runs": len(rs),
            "aaf_differs_from_b1": sum(r["aaf"] != r["b1"] for r in rs),
            "aaf_differs_from_b1_without_genuine_tool_result": sum(r["aaf"] != r["b1"] and not r["aaf_tool_used"] for r in rs),
            "b5_differs_from_b1": sum(r["b5"] != r["b1"] for r in rs),
            "b5_actions": dict(Counter(r["b5"] for r in rs).most_common(4)),
            "s6_correct_aaf": sum(r["aaf"] in (r["admissible"] or []) for r in rs if r["admissible"]),
            "s6_correct_b5": sum(r["b5"] in (r["admissible"] or []) for r in rs if r["admissible"]),
            "s6_correct_b1": sum(r["b1"] in (r["admissible"] or []) for r in rs if r["admissible"]),
            "s6_case_runs": sum(1 for r in rs if r["admissible"]),
        }
    (out / "containment_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
