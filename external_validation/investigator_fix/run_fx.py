"""Study 8 LLM arms (PROTOCOL.md). Fresh 36 cases: HY x3, HY2-L x3, B6 x1. Development: HY2-L x1 on the 96 earlier cases.
No-LLM arms (DET, AF-*, FETCH-ALL, HY2-F) are computed in analyze_fx.py."""
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
os.environ.setdefault("MAX_CALLS", "2200")
os.environ.setdefault("MAX_TOKENS", "1600000")
from revision_2026 import run_llm_arms as RA  # noqa: E402
from agentic_experiments.hybrid import run_hybrid  # noqa: E402
from agentic_experiments.hybrid_v2 import WideRuntimeMixin, run_hybrid_wide  # noqa: E402
from agentic_experiments.hybrid import run_agentic_only  # noqa: E402


class WideRuntime(WideRuntimeMixin, RA.Runtime):
    pass


def one(arm, fn, case, run, setname):
    try:
        if RA.BUDGET.stopped:
            raise RuntimeError(f"budget stop: {RA.BUDGET.stopped}")
        res = fn(case)
        if RA.BUDGET.stopped:
            raise RuntimeError(f"budget stop during case: {RA.BUDGET.stopped}")
        return {"set": setname, "arm": arm, "case": case["id"], "stratum": case["stratum"], "substratum": case.get("substratum"), "run": run,
                "action": res["selected_action"], **RA.slim(res), "trigger": (res.get("trigger") or {}).get("reasons"), "error": None}
    except Exception as exc:
        return {"set": setname, "arm": arm, "case": case["id"], "run": run, "action": None, "error": f"{type(exc).__name__}: {str(exc)[:200]}"}


def main() -> None:
    out = HERE / "results" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out.mkdir(parents=True, exist_ok=True)
    try:
        RA.Runtime("none")._call("Reply with a JSON object {\"ok\": true}.", {"ping": 1})
    except Exception as exc:
        (out / "STOPPED.txt").write_text(f"probe failed: {exc}")
        raise SystemExit(f"probe failed: {exc}")
    fresh = yaml.safe_load((HERE / "cases_fx_v1.yaml").read_text())["cases"]
    dev = [c for f in ("agentic_experiments/cases_v1.yaml", "agentic_experiments/cases_replication_v1.yaml", "external_validation/expert_study/cases_new_v1.yaml")
           for c in yaml.safe_load((ROOT / f).read_text())["cases"]]
    rt_none, rt_wide, rt_b6 = RA.Runtime("none"), WideRuntime("medium"), RA.Runtime("medium", RA.COORD_B6)
    plan = [("fresh", "HY2-L", lambda c: run_hybrid_wide(c, rt_wide), [1, 2, 3], fresh),
            ("fresh", "HY", lambda c: run_hybrid(c, rt_none), [1, 2, 3], fresh),
            ("fresh", "B6", lambda c: run_agentic_only(c, rt_b6), [1], fresh),
            ("dev", "HY2-L", lambda c: run_hybrid_wide(c, rt_wide), [1], dev)]
    for setname, arm, fn, runs, cases in plan:
        jobs = [(c, r) for r in runs for c in cases]
        with ThreadPoolExecutor(max_workers=6) as pool:
            rows = list(pool.map(lambda j: one(arm, fn, j[0], j[1], setname), jobs))
        (out / f"{setname}_{arm}.jsonl").write_text("\n".join(json.dumps(r, default=str) for r in rows))
        print(setname, arm, "errors", sum(1 for r in rows if r["error"]), Counter(r["action"] for r in rows).most_common(3),
              "calls", RA.BUDGET.calls, "tokens", RA.BUDGET.tokens, "stop", RA.BUDGET.stopped, flush=True)
    (out / "budget.json").write_text(json.dumps({"calls": RA.BUDGET.calls, "tokens": RA.BUDGET.tokens, "stopped": RA.BUDGET.stopped,
                                                 "caps": [RA.MAX_CALLS, RA.MAX_TOKENS], "model": RA.MODEL}, indent=1))


if __name__ == "__main__":
    main()
