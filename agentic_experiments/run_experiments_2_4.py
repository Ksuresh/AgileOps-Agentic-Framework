from __future__ import annotations

import csv
import hashlib
import json
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any

import yaml

from agentic_experiments.agentic_runtime import BoundedAgenticRuntime
from agentic_experiments.hybrid import deterministic_assessment, run_agentic_only, run_hybrid

ROOT = Path(__file__).resolve().parents[1]
DATASET = Path(__file__).with_name("cases_v1.yaml")
OUT = ROOT / "paper_results" / "agentic_experiments_2_4"
MODEL = os.getenv("OPENAI_MODEL", "gpt-5.6-luna")
REASONING = os.getenv("OPENAI_REASONING_EFFORT", "none")


def load_cases() -> list[dict[str, Any]]:
    doc = yaml.safe_load(DATASET.read_text())
    cases = list(doc["cases"])
    assert len(cases) == 32
    assert Counter(c["stratum"] for c in cases) == {"CLEAR": 8, "AMBIGUOUS": 8, "INCOMPLETE": 8, "MISLEADING": 8}
    return cases


def correct(action: str, case: dict[str, Any]) -> bool:
    return action in set(case["admissible_actions"])


def requested_tools(result: dict[str, Any]) -> list[str]:
    return [str(x.get("tool")) for x in result.get("tool_events", []) if x.get("tool")]


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"n_cases": len(rows), "by_stratum": {}}
    for stratum in ["CLEAR", "AMBIGUOUS", "INCOMPLETE", "MISLEADING"]:
        rs = [r for r in rows if r["stratum"] == stratum]
        out["by_stratum"][stratum] = {
            "n": len(rs),
            "deterministic_correct": sum(r["deterministic_correct"] for r in rs),
            "agentic_only_correct": sum(r["agentic_only_correct"] for r in rs),
            "hybrid_correct": sum(r["hybrid_correct"] for r in rs),
            "hybrid_invoked": sum(r["hybrid_invoked"] for r in rs),
            "agentic_only_tokens": sum(r["agentic_only_tokens"] for r in rs),
            "hybrid_tokens": sum(r["hybrid_tokens"] for r in rs),
        }
    ao_tokens = sum(r["agentic_only_tokens"] for r in rows)
    hy_tokens = sum(r["hybrid_tokens"] for r in rows)
    out["overall"] = {
        "deterministic_correct": sum(r["deterministic_correct"] for r in rows),
        "agentic_only_correct": sum(r["agentic_only_correct"] for r in rows),
        "hybrid_correct": sum(r["hybrid_correct"] for r in rows),
        "hybrid_invocation_rate": sum(r["hybrid_invoked"] for r in rows) / len(rows),
        "unnecessary_clear_invocations": sum(r["hybrid_invoked"] for r in rows if r["stratum"] == "CLEAR"),
        "agentic_only_tokens": ao_tokens,
        "hybrid_tokens": hy_tokens,
        "token_avoidance_vs_always_on": (1.0 - hy_tokens / ao_tokens) if ao_tokens else None,
        "governance_overrides": sum(r["governance_override"] for r in rows),
        "beneficial_overrides": sum(r["beneficial_override"] for r in rows),
    }
    incomplete = [r for r in rows if r["stratum"] == "INCOMPLETE"]
    out["experiment_3_incomplete"] = {
        "n": len(incomplete),
        "expected_tool_selected_agentic_only": sum(r["agentic_only_expected_tool"] for r in incomplete),
        "expected_tool_selected_hybrid": sum(r["hybrid_expected_tool"] for r in incomplete),
        "hybrid_pre_correct": sum(r["deterministic_correct"] for r in incomplete),
        "hybrid_post_correct": sum(r["hybrid_correct"] for r in incomplete),
        "hybrid_regrounding_success": sum((not r["deterministic_correct"]) and r["hybrid_correct"] for r in incomplete),
    }
    misleading = [r for r in rows if r["stratum"] == "MISLEADING"]
    out["experiment_4_misleading"] = {
        "n": len(misleading),
        "agentic_only_correct": sum(r["agentic_only_correct"] for r in misleading),
        "hybrid_correct": sum(r["hybrid_correct"] for r in misleading),
        "hybrid_governance_overrides": sum(r["governance_override"] for r in misleading),
        "beneficial_overrides": sum(r["beneficial_override"] for r in misleading),
        "agentic_only_evidence_refs_valid": sum(r["agentic_only_refs_valid"] for r in misleading),
        "hybrid_evidence_refs_valid": sum(r["hybrid_refs_valid"] for r in misleading),
    }
    lat = [r["hybrid_latency_ms"] for r in rows if r["hybrid_invoked"]]
    if lat:
        out["overall"]["hybrid_invoked_latency_ms_mean"] = mean(lat)
        out["overall"]["hybrid_invoked_latency_ms_median"] = median(lat)
    return out


def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required for the confirmatory Agentic run")
    cases = load_cases()
    OUT.mkdir(parents=True, exist_ok=True)
    runtime = BoundedAgenticRuntime(MODEL, REASONING)
    rows = []
    raw_records = []

    for case in cases:
        det = deterministic_assessment(case["evidence"])
        ao = run_agentic_only(case, runtime)
        hy = run_hybrid(case, runtime)
        det_ok = correct(det["selected_action"], case)
        ao_ok = correct(ao["selected_action"], case)
        hy_ok = correct(hy["selected_action"], case)
        expected_tool = case.get("expected_tool")
        ao_tools = requested_tools(ao)
        hy_tools = requested_tools(hy)
        override = bool(hy.get("governance_override", False))
        proposal_ok = correct(str(hy.get("agent_proposal", hy["selected_action"])), case)
        row = {
            "case_id": case["id"], "stratum": case["stratum"],
            "deterministic_action": det["selected_action"], "agentic_only_action": ao["selected_action"], "hybrid_action": hy["selected_action"],
            "deterministic_correct": det_ok, "agentic_only_correct": ao_ok, "hybrid_correct": hy_ok,
            "hybrid_invoked": bool(hy["agentic_invoked"]),
            "trigger_reasons": ";".join(hy["trigger"]["reasons"]),
            "agentic_only_tokens": int(ao["usage"]["total_tokens"]), "hybrid_tokens": int(hy["usage"]["total_tokens"]),
            "agentic_only_latency_ms": float(ao["latency_ms"]), "hybrid_latency_ms": float(hy["latency_ms"]),
            "agentic_only_refs_valid": bool(ao.get("all_evidence_refs_valid", False)), "hybrid_refs_valid": bool(hy.get("all_evidence_refs_valid", True)),
            "agentic_only_schema_valid": bool(ao.get("all_schema_valid", False)), "hybrid_schema_valid": bool(hy.get("all_schema_valid", True)),
            "agentic_only_expected_tool": bool(expected_tool and expected_tool in ao_tools),
            "hybrid_expected_tool": bool(expected_tool and expected_tool in hy_tools),
            "governance_override": override,
            "beneficial_override": bool(override and (not proposal_ok) and hy_ok),
        }
        rows.append(row)
        raw_records.append({"case_id": case["id"], "deterministic": det, "agentic_only": ao, "hybrid": hy})
        print(case["id"], case["stratum"], det["selected_action"], ao["selected_action"], hy["selected_action"])

    dataset_sha = hashlib.sha256(DATASET.read_bytes()).hexdigest()
    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model": MODEL, "reasoning_effort": REASONING,
        "dataset": str(DATASET.relative_to(ROOT)), "dataset_sha256": dataset_sha,
        "n_cases": len(cases),
    }
    (OUT / "metadata.json").write_text(json.dumps(metadata, indent=2))
    (OUT / "raw_outputs.json").write_text(json.dumps(raw_records, indent=2, ensure_ascii=False))
    (OUT / "summary.json").write_text(json.dumps(summarize(rows), indent=2))
    with (OUT / "case_results.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    print(json.dumps(summarize(rows), indent=2))


if __name__ == "__main__":
    main()
