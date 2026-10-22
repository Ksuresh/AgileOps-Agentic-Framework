"""LLM arms of the expert-labelled extension study (PROTOCOL.md). Runs only on the 48 new cases.

HY  Hybrid AAF (frozen hybrid.run_hybrid, reasoning none)            runs 1-3
AO  Agentic-only original (frozen run_agentic_only, reasoning none)   run 1
B6  enhanced Agentic-only (reasoning medium + governance rules)       run 1
Scoring is done separately (analyze.py) against the expert labels committed before this script is run.
Reuses the budget-capped runtime of revision_2026/run_llm_arms.py (probe call first; stop on account errors).
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MAX_CALLS", "1500")
os.environ.setdefault("MAX_TOKENS", "1200000")
from revision_2026 import run_llm_arms as RA  # noqa: E402  (reads MAX_CALLS / MAX_TOKENS at import)
from agentic_experiments.hybrid import run_agentic_only, run_hybrid  # noqa: E402

LABELS = HERE / "expert_labels.json"


def one(arm, fn, case, run):
    try:
        if RA.BUDGET.stopped:
            raise RuntimeError(f"budget stop: {RA.BUDGET.stopped}")
        res = fn(case)
        if RA.BUDGET.stopped:
            raise RuntimeError(f"budget stop during case: {RA.BUDGET.stopped}")
        return {"arm": arm, "case": case["id"], "stratum": case["stratum"], "substratum": case.get("substratum"), "run": run,
                "action": res["selected_action"], **RA.slim(res), "trigger": (res.get("trigger") or {}).get("reasons"), "error": None}
    except Exception as exc:
        return {"arm": arm, "case": case["id"], "stratum": case["stratum"], "substratum": case.get("substratum"), "run": run,
                "action": None, "error": f"{type(exc).__name__}: {str(exc)[:200]}"}


def main() -> None:
    if not LABELS.exists():
        raise SystemExit("expert_labels.json must be committed before any run (PROTOCOL.md)")
    out = HERE / "results" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out.mkdir(parents=True, exist_ok=True)
    probe = RA.Runtime("none")
    try:
        probe._call("Reply with a JSON object {\"ok\": true}.", {"ping": 1})
    except Exception as exc:
        (out / "STOPPED.txt").write_text(f"probe failed: {exc}")
        raise SystemExit(f"probe failed: {exc}")
    cases = yaml.safe_load((HERE / "cases_new_v1.yaml").read_text())["cases"]
    rt_none, rt_b6 = RA.Runtime("none"), RA.Runtime("medium", RA.COORD_B6)
    plan = [("HY", lambda c: run_hybrid(c, rt_none), [1, 2, 3]), ("AO", lambda c: run_agentic_only(c, rt_none), [1]),
            ("B6", lambda c: run_agentic_only(c, rt_b6), [1])]
    for arm, fn, runs in plan:
        jobs = [(c, r) for r in runs for c in cases]
        with ThreadPoolExecutor(max_workers=6) as pool:
            rows = list(pool.map(lambda j: one(arm, fn, *j), jobs))
        (out / f"{arm}.jsonl").write_text("\n".join(json.dumps(r, default=str) for r in rows))
        print(arm, "errors", sum(1 for r in rows if r["error"]), Counter(r["action"] for r in rows).most_common(4),
              "calls", RA.BUDGET.calls, "tokens", RA.BUDGET.tokens, "stop", RA.BUDGET.stopped, flush=True)
    (out / "budget.json").write_text(json.dumps({"calls": RA.BUDGET.calls, "tokens": RA.BUDGET.tokens, "stopped": RA.BUDGET.stopped,
                                                 "caps": [RA.MAX_CALLS, RA.MAX_TOKENS], "model": RA.MODEL}, indent=1))


if __name__ == "__main__":
    main()
