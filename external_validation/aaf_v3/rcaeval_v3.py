"""AAF v3, telemetry side (PLAN.md change 1): change-aware materiality. Designed on RCAEval RE2 only.

v1 maps a metric into AAF only when its absolute level crosses AAF's fixed thresholds. v3 additionally treats a
signal as material when it departs from the same system's own pre-window baseline by a fixed ratio, and maps that
departure onto AAF's canonical *elevated* level for the field (the same normalisation used for qualitative text facts):
  largest service latency-90 ratio >= 4x                      -> sre.p95_latency_ms   = 450 (elevated), if lower/absent
  largest CPU ratio >= 4x  or  largest memory ratio >= 1.5x    -> sre.saturation_pct   = 85  (elevated), if lower/absent
Ratios are the adapter's own label-blind largest post/pre ratios (5-minute windows). Thresholds chosen on RE2
(fault vs control windows), rounded towards the conservative side. Decision path, weights and policy: unchanged.
"""
from __future__ import annotations

import copy
from typing import Any

import pandas as pd

from external_validation.rcaeval.adapter import adapt
from orchestrator.cross_domain_v2 import apply_interaction_policy_v2
from orchestrator.utility import choose_action_details

WEIGHTS = (0.45, 0.20, 0.35)  # frozen RCAEval configuration
LAT_RATIO, CPU_RATIO, MEM_RATIO = 4.0, 4.0, 1.5
P95_ELEVATED, SAT_ELEVATED = 450.0, 85.0


def _ratio(d: dict[str, Any] | None) -> float | None:
    return None if not d or d.get("ratio") is None else float(d["ratio"])


def decide(telemetry: dict[str, Any]) -> str:
    base = choose_action_details(telemetry, WEIGHTS)
    return str(apply_interaction_policy_v2(telemetry, str(base["selected_action"]))["selected_action"])


def v3_telemetry(telemetry: dict[str, Any], diag: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    tel = copy.deepcopy(telemetry)
    sre = tel.setdefault("sre", {"_evidence": {}})
    lat = _ratio(diag.get("latency90_largest_ratio"))
    cpu = _ratio(diag.get("cpu_largest_ratio"))
    mem = _ratio((diag.get("auxiliary_evidence") or {}).get("memory_largest_ratio"))
    fired = {}
    if lat is not None and lat >= LAT_RATIO and float(sre.get("p95_latency_ms") or 0) < P95_ELEVATED:
        sre["p95_latency_ms"] = P95_ELEVATED
        sre["_evidence"]["p95_latency_ms"] = {"status": "proxy", "source": f"relative: latency-90 {lat:.2f}x own baseline"}
        fired["latency"] = lat
    if ((cpu is not None and cpu >= CPU_RATIO) or (mem is not None and mem >= MEM_RATIO)) and float(sre.get("saturation_pct") or 0) < SAT_ELEVATED:
        sre["saturation_pct"] = SAT_ELEVATED
        sre["_evidence"]["saturation_pct"] = {"status": "proxy", "source": f"relative: cpu {cpu}x / memory {mem}x own baseline"}
        fired["resource"] = {"cpu": cpu, "mem": mem}
    return tel, fired


def decisions(metrics: pd.DataFrame, t: int) -> dict[str, Any]:
    telemetry, diag = adapt(metrics, t, logs=None)
    v1 = decide(telemetry)
    tel3, fired = v3_telemetry(telemetry, diag)
    return {"v1": v1, "v3": decide(tel3), "v3_relative_fired": fired, "diag": diag}
