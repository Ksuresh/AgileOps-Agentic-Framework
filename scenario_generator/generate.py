from __future__ import annotations
import json, random
from typing import Dict, List, Any

SCENARIO_TYPES = [
  ("Deployment failures", 4),
  ("Cost anomalies", 4),
  ("Security violations", 4),
  ("Reliability degradations", 4),
  ("Compliance preparation", 3),
  ("Resource scaling", 3),
  ("Incident review", 3),
  ("Policy propagation", 2),
  ("Cost–performance simulation", 2),
  ("Anomaly-to-insight", 1),
]

def _jitter(val: float, pct: float) -> float:
    return val * (1.0 + random.uniform(-pct, pct))

def generate_scenarios(seed: int, noise: Dict[str, Any]) -> List[Dict[str, Any]]:
    random.seed(seed)
    missing_p = float(noise.get("missing_evidence_prob", 0.0))
    contra_p = float(noise.get("contradiction_prob", 0.0))
    jitter_pct = float(noise.get("metric_jitter_pct", 0.0))

    scenarios: List[Dict[str, Any]] = []
    idx = 1
    for s_type, count in SCENARIO_TYPES:
        for _ in range(count):
            base = {
                "incident_id": f"SC-{idx:02d}",
                "scenario_type": s_type,
                "telemetry": {
                    "deploy": {
                        "restart_loops": int(_jitter(random.choice([0, 6, 12, 18]), jitter_pct)),
                        "config_drift": random.random() < 0.35,
                        "pipeline_failed": random.random() < 0.30,
                    },
                    "sre": {
                        "p95_latency_ms": float(_jitter(random.choice([180, 220, 450, 900]), jitter_pct)),
                        "error_rate_pct": float(_jitter(random.choice([0.5, 2.0, 8.0, 14.0]), jitter_pct)),
                        "saturation_pct": float(_jitter(random.choice([55, 70, 85, 92]), jitter_pct)),
                    },
                    "finops": {
                        "cost_spike_pct": float(_jitter(random.choice([0, 8, 22, 40]), jitter_pct)),
                        "hpa_scale_to": int(_jitter(random.choice([4, 7, 11, 16]), jitter_pct)),
                    },
                    "sec": {
                        "critical_cves": random.choice([0, 0, 1, 2]),
                        "policy_violation": random.random() < 0.18,
                        "iam_drift": random.random() < 0.12,
                    },
                },
                "ground_truth": {"primary_domain": None, "recommended_action": None}
            }

            t = base["telemetry"]
            if t["sec"]["critical_cves"] > 0 or t["sec"]["policy_violation"] or t["sec"]["iam_drift"]:
                base["ground_truth"]["primary_domain"] = "DevSecOps"
                base["ground_truth"]["recommended_action"] = "Patch or block release"
            elif t["sre"]["p95_latency_ms"] >= 450 or t["sre"]["error_rate_pct"] >= 8.0:
                base["ground_truth"]["primary_domain"] = "SRE"
                base["ground_truth"]["recommended_action"] = "Rollback or mitigate"
            elif t["finops"]["cost_spike_pct"] >= 22:
                base["ground_truth"]["primary_domain"] = "FinOps"
                base["ground_truth"]["recommended_action"] = "Scale adjustment"
            elif t["deploy"]["pipeline_failed"] or t["deploy"]["config_drift"] or t["deploy"]["restart_loops"] >= 12:
                base["ground_truth"]["primary_domain"] = "DevOps"
                base["ground_truth"]["recommended_action"] = "Rollback to stable build"
            else:
                base["ground_truth"]["primary_domain"] = "DevOps"
                base["ground_truth"]["recommended_action"] = "Monitor"

            if random.random() < missing_p:
                drop = random.choice(["deploy","sre","finops","sec"])
                base["telemetry"][drop]["_missing"] = True

            if random.random() < contra_p:
                base["telemetry"]["sre"]["error_rate_pct"] = float(_jitter(random.choice([0.5, 2.0]), jitter_pct))

            scenarios.append(base)
            idx += 1

    return scenarios

def save_scenarios(path: str, scenarios: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(scenarios, f, indent=2)
