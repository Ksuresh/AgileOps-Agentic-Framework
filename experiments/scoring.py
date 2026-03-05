"""Scoring helpers for the synthetic evaluation.

These metrics are designed to mirror the paper's reporting needs, while
remaining fully reproducible.
"""

from __future__ import annotations

from typing import Dict, Any, List
import math


def score_primary_domain_accuracy(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    """Accuracy of predicted primary domain vs scenario ground truth."""
    correct = 0
    total = 0
    for r in rows:
        gt = (r.get("ground_truth") or {}).get("primary_domain")
        pred = r.get("predicted_primary_domain")
        if gt is None or pred is None:
            continue
        total += 1
        if str(gt).lower() == str(pred).lower():
            correct += 1
    acc = (correct / total) if total else 0.0
    # A tiny, conservative estimate of dispersion for reporting
    # (binomial std dev). Keep it simple and deterministic.
    std = math.sqrt(acc * (1 - acc) / total) if total else 0.0
    return {"accuracy": acc, "n": float(total), "std": std}


def compute_rar_stats(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    triggered = sum(1 for r in rows if r.get("rar", {}).get("triggered"))
    unresolved = sum(1 for r in rows if r.get("rar", {}).get("triggered") and not r.get("rar", {}).get("accepted"))
    n = len(rows)
    return {
        "triggered": float(triggered),
        "unresolved": float(unresolved),
        "trigger_rate": float(triggered / n) if n else 0.0,
    }


def _stats(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0}
    vs = sorted(values)
    n = len(vs)
    mean = sum(vs) / n
    var = sum((x - mean) ** 2 for x in vs) / n
    std = math.sqrt(var)
    def pct(p: float) -> float:
        idx = min(n - 1, max(0, int(math.ceil(p * n) - 1)))
        return vs[idx]
    return {"mean": mean, "std": std, "p50": pct(0.50), "p95": pct(0.95), "p99": pct(0.99)}


def compute_latency_stats(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    totals = [float(r.get("timings", {}).get("TOTAL", 0.0)) for r in rows]
    no_rar = [float(r.get("timings", {}).get("TOTAL", 0.0)) for r in rows if not r.get("rar", {}).get("triggered")]
    with_rar = [float(r.get("timings", {}).get("TOTAL", 0.0)) for r in rows if r.get("rar", {}).get("triggered")]
    return {"total": _stats(totals), "no_rar": _stats(no_rar), "with_rar": _stats(with_rar)}


def compute_utility_stats(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    us = [float(r.get("utility", {}).get("best_utility", 0.0)) for r in rows]
    st = _stats(us)
    return {"mean": st["mean"], "std": st["std"]}


def compute_xi_stats(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    xs = [float(r.get("explainability", {}).get("xi", 0.0)) for r in rows]
    st = _stats(xs)
    return {"mean": st["mean"], "std": st["std"]}
