"""Text-evidence development pilot: does an LLM reader recover decision-relevant
evidence from messy text better than a regex reader, with the frozen deterministic
AAF rules keeping final authority?

Arms (same frozen rules decide in every arm):
  T0_text_blind   rules on structured evidence only (text ignored)
  T1_regex        regex reader -> validator -> rules
  T2_llm          LLM reader   -> validator -> rules   (run --repeats times; case correct if majority correct)
  CEILING         rules on structured evidence + ground-truth text values (reference decision)

Readers may only fill fields that are missing in the structured evidence; they never
overwrite measured telemetry and never propose actions.

Usage (from repo root):
  python -m text_evidence_pilot.run_pilot --mock                 # offline dry run, no API key
  OPENAI_API_KEY=... python -m text_evidence_pilot.run_pilot     # real pilot (gpt-5.6-luna by default)
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import platform
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from scipy.stats import binomtest

from agentic_experiments.hybrid import deterministic_assessment
from text_evidence_pilot import regex_reader
from text_evidence_pilot.llm_reader import MockReader, OpenAIReader, READER_SYSTEM
from text_evidence_pilot.schema import FIELD_SPECS, split
from text_evidence_pilot.validator import check

HERE = Path(__file__).resolve().parent
CASES = HERE / "cases_text_pilot_v0.yaml"


def missing_in_structured(case: dict[str, Any]) -> set[str]:
    out = set()
    for block, fields in case["structured"].items():
        meta = (fields or {}).get("_evidence", {}) or {}
        for name, m in meta.items():
            if isinstance(m, dict) and m.get("status") == "missing":
                out.add(f"{block}.{name}")
    return out


def fill(case: dict[str, Any], extractions: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate extractions and write accepted ones into the missing fields only."""
    ev = copy.deepcopy(case["structured"])
    gaps = missing_in_structured(case)
    accepted, rejected, ignored = {}, [], []
    for x in extractions:
        ok, reason = check(x, case["documents"])
        if not ok:
            rejected.append({"field": x.get("field"), "reason": reason, "value": x.get("value")})
            continue
        f = x["field"]
        if f not in gaps:
            ignored.append(f)  # never overwrite structured telemetry
            continue
        block, name = split(f)
        v = bool(x["value"]) if FIELD_SPECS[f]["type"] == "bool" else float(x["value"])
        dst = ev.setdefault(block, {})
        dst[name] = v
        dst.setdefault("_evidence", {})[name] = {"status": "measured", "source": f"{x['doc_id']}:text"}
        accepted[f] = v
    return ev, {"accepted": accepted, "rejected": rejected, "ignored_present_fields": ignored}


def decide(evidence: dict[str, Any]) -> str:
    return str(deterministic_assessment(copy.deepcopy(evidence))["selected_action"])


def ceiling_evidence(case: dict[str, Any]) -> dict[str, Any]:
    ev = copy.deepcopy(case["structured"])
    for f, t in (case.get("truth") or {}).items():
        block, name = split(f)
        ev.setdefault(block, {})[name] = t["value"]
        ev[block].setdefault("_evidence", {})[name] = {"status": "measured", "source": "truth"}
    return ev


def field_score(case: dict[str, Any], accepted: dict[str, Any]) -> dict[str, int]:
    truth = case.get("truth") or {}
    s = Counter()
    for f, t in truth.items():
        if f not in accepted:
            s["missed"] += 1
        elif FIELD_SPECS[f]["type"] == "bool":
            s["correct" if accepted[f] == t["value"] else "wrong"] += 1
        else:
            tv = float(t["value"])
            s["correct" if abs(accepted[f] - tv) <= max(1e-6, 0.01 * abs(tv)) else "wrong"] += 1
    s["spurious"] += sum(1 for f in accepted if f not in truth)
    s["truth_fields"] += len(truth)
    return dict(s)


def regex_extractions(case: dict[str, Any]) -> list[dict[str, Any]]:
    return [{"field": f, "value": r["value"], "doc_id": r["doc_id"], "quote": r["quote"], "method": "literal"}
            for f, r in regex_reader.read(case["documents"]).items()]


def mcnemar(a: list[bool], b: list[bool]) -> dict[str, Any]:
    n10 = sum(1 for x, y in zip(a, b) if x and not y)
    n01 = sum(1 for x, y in zip(a, b) if y and not x)
    p = binomtest(min(n10, n01), n10 + n01, 0.5).pvalue if n10 + n01 else 1.0
    return {"a_only": n10, "b_only": n01, "exact_p": round(float(p), 4)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", action="store_true", help="offline dry run with the mock reader")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--out", default="text_evidence_pilot/results")
    args = ap.parse_args()

    raw_cases = CASES.read_bytes()
    cases = yaml.safe_load(raw_cases)["cases"]
    reader = MockReader() if args.mock else OpenAIReader()
    out = Path(args.out) / ("mock" if args.mock else datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    out.mkdir(parents=True, exist_ok=True)

    rows, raw_log = [], []
    tot_fields = {arm: Counter() for arm in ("T1_regex", "T2_llm")}
    reject_reasons = Counter()
    for case in cases:
        ceiling = decide(ceiling_evidence(case))
        t0 = decide(case["structured"])

        ev1, info1 = fill(case, regex_extractions(case))
        t1 = decide(ev1)
        fs1 = field_score(case, info1["accepted"]); tot_fields["T1_regex"].update(fs1)

        runs = []
        for r in range(args.repeats):
            res = reader.read(case)
            ev2, info2 = fill(case, res["extractions"])
            act = decide(ev2)
            fs2 = field_score(case, info2["accepted"]); tot_fields["T2_llm"].update(fs2)
            reject_reasons.update(x["reason"] for x in info2["rejected"])
            runs.append({"action": act, "correct": act == ceiling, "fields": fs2, "rejected": len(info2["rejected"]),
                         "usage": res["usage"], "latency_ms": res["latency_ms"], "error": res["error"]})
            raw_log.append({"case": case["id"], "run": r, "reader_output": res, "validation": info2, "action": act})

        n_ok = sum(x["correct"] for x in runs)
        rows.append({
            "case": case["id"], "tier": case["tier"], "ceiling_action": ceiling,
            "T0_action": t0, "T0_correct": t0 == ceiling,
            "T1_action": t1, "T1_correct": t1 == ceiling, "T1_fields": json.dumps(fs1, sort_keys=True),
            "T2_runs_correct": f"{n_ok}/{args.repeats}", "T2_correct": n_ok * 2 > args.repeats,
            "T2_actions": json.dumps(Counter(x["action"] for x in runs)),
            "T2_rejected_total": sum(x["rejected"] for x in runs),
            "T2_errors": sum(1 for x in runs if x["error"]),
            "T2_tokens_mean": round(sum(x["usage"]["total_tokens"] for x in runs) / max(1, len(runs)), 1),
            "T2_latency_ms_mean": round(sum(x["latency_ms"] for x in runs) / max(1, len(runs)), 1),
        })

    by_tier = defaultdict(lambda: Counter())
    for r in rows:
        t = by_tier[r["tier"]]; t["n"] += 1
        for arm in ("T0", "T1", "T2"):
            t[arm] += int(r[f"{arm}_correct"])
    summary = {
        "n_cases": len(rows), "repeats": args.repeats, "mock": args.mock,
        "decision_agreement_with_ceiling": {arm: sum(int(r[f"{arm}_correct"]) for r in rows) for arm in ("T0", "T1", "T2")},
        "by_tier": {k: dict(v) for k, v in by_tier.items()},
        "field_level": {k: dict(v) for k, v in tot_fields.items()},
        "llm_rejections_by_reason": dict(reject_reasons),
        "paired_T2_vs_T1": mcnemar([r["T2_correct"] for r in rows], [r["T1_correct"] for r in rows]),
        "llm_tokens_total": sum(x["reader_output"]["usage"]["total_tokens"] for x in raw_log),
        "llm_failed_calls": sum(1 for x in raw_log if x["reader_output"]["error"]),
    }
    meta = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(),
        "cases_sha256": hashlib.sha256(raw_cases).hexdigest(),
        "reader_prompt_sha256": hashlib.sha256(READER_SYSTEM.encode()).hexdigest(),
        "model": None if args.mock else os.environ.get("OPENAI_MODEL", "gpt-5.6-luna"),
        "reasoning_effort": None if args.mock else os.environ.get("OPENAI_REASONING_EFFORT", "none"),
        "github_run_id": os.environ.get("GITHUB_RUN_ID"), "git_sha": os.environ.get("GITHUB_SHA"),
    }

    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    (out / "metadata.json").write_text(json.dumps(meta, indent=2))
    with (out / "raw_runs.jsonl").open("w") as fh:
        for rec in raw_log:
            fh.write(json.dumps(rec, default=str) + "\n")
    with (out / "case_results.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(json.dumps(summary, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
