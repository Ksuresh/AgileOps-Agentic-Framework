"""Scoring helpers for AAF evaluation."""

from __future__ import annotations

from typing import Dict, Any, List
import math


def score_primary_domain_accuracy(rows: List[Dict[str, Any]]) -> Dict[str, float]:
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
    std = math.sqrt(acc * (1 - acc) / total) if total else 0.0
    return {"accuracy": acc, "n": float(total), "std": std}


def score_action_match(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    correct = 0
    total = 0

    for r in rows:
        gt = (r.get("ground_truth") or {}).get("expected_action") or (r.get("ground_truth") or {}).get("recommended_action")
        pred = (r.get("utility") or {}).get("selected_action") or r.get("recommended_action")

        if gt is None or pred is None:
            continue

        total += 1
        if _normalize_action(gt) == _normalize_action(pred):
            correct += 1

    rate = (correct / total) if total else 0.0
    std = math.sqrt(rate * (1 - rate) / total) if total else 0.0
    return {"action_match_rate": rate, "n": float(total), "std": std}


def _normalize_action(action: str) -> str:
    a = str(action).lower().strip()

    if "rollback" in a:
        return "rollback"
    if "patch" in a or "block" in a or "security" in a:
        return "patch_block"
    if "scale" in a or "scaling" in a:
        return "scale"
    if "mitigate" in a or "monitor" in a:
        return "mitigate_monitor"
    if "review" in a:
        return "review"
    if "observe" in a or "no action" in a:
        return "observe"

    return a


def compute_rar_stats(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    n = len(rows)
    triggered = 0
    accepted = 0
    unresolved = 0
    improvements = []

    for r in rows:
        rar = r.get("rar", {}) or {}
        if rar.get("triggered"):
            triggered += 1
            if rar.get("accepted"):
                accepted += 1
            else:
                unresolved += 1

            before = float(rar.get("before", 0.0) or 0.0)
            after = float(rar.get("after", before) or before)
            improvements.append(after - before)

    avg_improvement = sum(improvements) / len(improvements) if improvements else 0.0

    return {
        "triggered": float(triggered),
        "accepted": float(accepted),
        "unresolved": float(unresolved),
        "trigger_rate": float(triggered / n) if n else 0.0,
        "acceptance_rate_when_triggered": float(accepted / triggered) if triggered else 0.0,
        "avg_consensus_improvement": float(avg_improvement),
    }


def compute_consensus_stats(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    values = [float(r.get("consensus_score", 0.0) or 0.0) for r in rows]
    st = _stats(values)
    return {"mean": st["mean"], "std": st["std"], "p50": st["p50"], "p95": st["p95"]}


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
    values = [float(r.get("utility", {}).get("best_utility", 0.0)) for r in rows]
    st = _stats(values)
    return {"mean": st["mean"], "std": st["std"], "p50": st["p50"], "p95": st["p95"]}


def compute_xi_stats(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    values = [float(r.get("explainability", {}).get("xi", 0.0)) for r in rows]
    readability = [float(r.get("explainability", {}).get("readability", 0.0)) for r in rows]
    evidence = [float(r.get("explainability", {}).get("evidence_clarity", 0.0)) for r in rows]
    trace = [float(r.get("explainability", {}).get("traceability", 0.0)) for r in rows]

    xi = _stats(values)
    rd = _stats(readability)
    ev = _stats(evidence)
    tr = _stats(trace)

    return {
        "mean": xi["mean"],
        "std": xi["std"],
        "readability_mean": rd["mean"],
        "evidence_clarity_mean": ev["mean"],
        "traceability_mean": tr["mean"],
    }
