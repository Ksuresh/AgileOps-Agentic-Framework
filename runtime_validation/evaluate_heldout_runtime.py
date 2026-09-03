from __future__ import annotations

"""Evaluate frozen held-out Sock Shop cases using preserved runtime evidence.

Scientific safeguards:
- single-point max-container CPU remains contextual proxy evidence and cannot by
  itself establish SRE materiality;
- restart materiality uses observed temporal transitions, not intervention labels;
- AAF-full, no-interaction, and dominant-domain baseline consume the same telemetry.
"""

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from orchestrator.cross_domain import apply_interaction_policy
from orchestrator.decision_baselines import choose_dominant_domain_action
from orchestrator.utility import choose_action_details
from runtime_validation.evidence_adapter import write_evidence

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = Path(__file__).with_name("interventions.yaml")


def load_manifest() -> Dict[str, Dict[str, Any]]:
    data = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    return {case["id"]: case for case in data["cases"]}


def latest_artifact(case_id: str, repetition: int) -> Optional[Path]:
    base = ROOT / "runtime_validation" / "artifacts" / case_id / f"rep-{repetition}"
    dirs = sorted([p for p in base.glob("*") if p.is_dir()]) if base.exists() else []
    return dirs[-1] if dirs else None


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _parse_ts(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def normalize_runtime_evidence(case_dir: Path, telemetry: Dict[str, Any]) -> Dict[str, Any]:
    # A one-shot maximum CPU value is preserved in raw observables but is not a
    # time-window SRE measurement. This removes the RT-01 false-positive mode.
    sre = telemetry.setdefault("sre", {})
    sre_meta = sre.setdefault("_evidence", {})
    if "saturation_pct" in sre_meta and str(sre_meta["saturation_pct"].get("status", "")) == "proxy":
        sre_meta["saturation_pct"] = {
            "status": "missing",
            "source": str(sre_meta["saturation_pct"].get("source", "docker stats max CPU percentage")),
            "note": "Single-point max-container CPU retained as contextual runtime observable; not sufficient alone for SRE materiality.",
        }

    load = _load_json(case_dir / "load_observation.json")
    if load:
        if load.get("latency_p95_ms") is not None:
            sre["p95_latency_ms"] = float(load["latency_p95_ms"])
            sre_meta["p95_latency_ms"] = {
                "status": "measured",
                "source": str(load.get("source", "direct HTTP load observation")),
                "note": "Measured during active request load.",
            }
        if load.get("error_rate_pct") is not None:
            sre["error_rate_pct"] = float(load["error_rate_pct"])
            sre_meta["error_rate_pct"] = {
                "status": "measured",
                "source": str(load.get("source", "direct HTTP load observation")),
                "note": "Measured HTTP request failure percentage.",
            }
        telemetry.setdefault("_runtime_observables", {})["load_observation"] = load

    temporal = _load_json(case_dir / "temporal_process_observation.json")
    if temporal:
        restarts = int(temporal.get("observed_restart_events", 0) or 0)
        samples = [str(x) for x in temporal.get("started_at_samples", []) if x]
        deploy = telemetry.setdefault("deploy", {})
        meta = deploy.setdefault("_evidence", {})
        deploy["restart_loops"] = restarts
        meta["restart_loops"] = {
            "status": "measured",
            "source": str(temporal.get("source", "temporal Docker process observation")),
            "note": "Observed Docker State.StartedAt transitions.",
        }
        if restarts >= 2 and len(samples) >= restarts + 1:
            post = samples[-restarts:]
            window = max(0.0, (_parse_ts(post[-1]) - _parse_ts(post[0])).total_seconds())
            deploy["restart_burst_count"] = restarts
            deploy["restart_window_seconds"] = window
            meta["restart_burst_count"] = {
                "status": "measured", "source": str(temporal.get("source", "temporal Docker process observation")),
                "note": "Count of observed process transitions in the post-intervention burst."
            }
            meta["restart_window_seconds"] = {
                "status": "measured", "source": str(temporal.get("source", "temporal Docker process observation")),
                "note": "Elapsed time between first and last observed post-intervention transition."
            }
        telemetry.setdefault("_runtime_observables", {})["temporal_process_observation"] = temporal
    return telemetry


def evaluate_case(case_id: str, rep: int, baseline_dir: Path, manifest: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    case_dir = latest_artifact(case_id, rep)
    if case_dir is None:
        raise FileNotFoundError(f"No artifact for {case_id} rep {rep}")
    t = write_evidence(case_dir, baseline_dir=baseline_dir)
    t = normalize_runtime_evidence(case_dir, t)
    (case_dir / "heldout_telemetry_evaluated.json").write_text(json.dumps(t, indent=2), encoding="utf-8")

    oracle = manifest[case_id]
    admissible = set(oracle.get("admissible_actions", []))
    baseline_action, baseline_severity, baseline_domain = choose_dominant_domain_action(t)
    no_interaction = choose_action_details(t, (0.4, 0.3, 0.3))["selected_action"]
    full = apply_interaction_policy(t, no_interaction)
    full_action = full["selected_action"]
    dominant = full["interaction_state"].get("dominant_interaction")

    return {
        "case_id": case_id,
        "repetition": rep,
        "oracle_actions": json.dumps(sorted(admissible)),
        "baseline_action": baseline_action,
        "baseline_match": baseline_action in admissible,
        "baseline_domain": baseline_domain,
        "baseline_severity": round(float(baseline_severity), 4),
        "aaf_no_interaction_action": no_interaction,
        "aaf_no_interaction_match": no_interaction in admissible,
        "aaf_full_action": full_action,
        "aaf_full_match": full_action in admissible,
        "interaction_applied": bool(full["interaction_applied"]),
        "dominant_interaction": None if dominant is None else dominant["name"],
        "restart_burst_count": t.get("deploy", {}).get("restart_burst_count"),
        "restart_window_seconds": t.get("deploy", {}).get("restart_window_seconds"),
        "p95_latency_ms": t.get("sre", {}).get("p95_latency_ms"),
        "error_rate_pct": t.get("sre", {}).get("error_rate_pct"),
        "resource_footprint_proxy_pct": t.get("finops", {}).get("cost_spike_pct"),
    }


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    return {
        "n": n,
        "runtime_system": "Sock Shop",
        "evaluation_reference": "frozen held-out intervention oracle",
        "dominant_domain_baseline_action_oracle_agreement": sum(bool(r["baseline_match"]) for r in rows) / n,
        "aaf_no_interaction_action_oracle_agreement": sum(bool(r["aaf_no_interaction_match"]) for r in rows) / n,
        "aaf_full_action_oracle_agreement": sum(bool(r["aaf_full_match"]) for r in rows) / n,
        "interpretation": "Held-out benchmark-runtime intervention-oracle agreement using preserved direct/proxy evidence; not production accuracy.",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="HRT-01,HRT-02,HRT-03,HRT-04,HRT-05")
    ap.add_argument("--repetitions", default="1,2,3")
    ap.add_argument("--out", default="results_heldout_runtime")
    args = ap.parse_args()
    cases = [x.strip() for x in args.cases.split(",") if x.strip()]
    reps = [int(x.strip()) for x in args.repetitions.split(",") if x.strip()]
    manifest = load_manifest()
    baseline = latest_artifact("HRT-01", reps[0])
    if baseline is None:
        raise FileNotFoundError("HRT-01 baseline artifact required")
    rows = [evaluate_case(c, r, baseline, manifest) for c in cases for r in reps]
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    with (out / "runtime_case_results.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    summary = summarize(rows)
    (out / "runtime_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
