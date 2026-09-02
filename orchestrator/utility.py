from __future__ import annotations

from typing import Any, Dict, Tuple

from orchestrator.severity import severity_scores


def utility_score(performance_score: float, cost_efficiency_score: float, risk_reduction_score: float, w: Tuple[float, float, float]) -> float:
    w_perf, w_cost, w_risk = w
    return w_perf * performance_score + w_cost * cost_efficiency_score + w_risk * risk_reduction_score


def _action_components(telemetry: Dict[str, Any]) -> Dict[str, Tuple[float, float, float]]:
    s = severity_scores(telemetry)
    deployment, reliability, cost, security = s["deployment"], s["reliability"], s["cost"], s["security"]
    return {
        "Rollback to stable deployment": (min(1.0, 0.35 + 0.40 * deployment + 0.25 * reliability), 0.65, min(1.0, 0.35 + 0.35 * deployment + 0.10 * security)),
        "Block release and fix pipeline": (min(1.0, 0.25 + 0.45 * deployment), 0.75, min(1.0, 0.40 + 0.30 * deployment + 0.20 * security)),
        "Mitigate and monitor": (min(1.0, 0.30 + 0.35 * reliability), 0.68, min(1.0, 0.30 + 0.20 * reliability)),
        "Scale adjustment": (min(1.0, 0.32 + 0.38 * reliability + 0.28 * cost), min(1.0, 0.48 + 0.40 * cost), min(1.0, 0.32 + 0.25 * reliability)),
        "Review scaling policy": (min(1.0, 0.22 + 0.25 * cost), min(1.0, 0.60 + 0.35 * cost), min(1.0, 0.32 + 0.18 * cost)),
        "Patch or block release": (min(1.0, 0.22 + 0.25 * security), 0.70, min(1.0, 0.48 + 0.48 * security)),
        "No action (observe)": (0.20, 0.95, max(0.05, 0.30 - 0.20 * max(deployment, reliability, cost, security))),
    }


def _action_fit_bonus(action: str, s: Dict[str, float]) -> float:
    deployment, reliability, cost, security = s["deployment"], s["reliability"], s["cost"], s["security"]
    dominant = max(s, key=s.get)
    bonus = 0.0
    if dominant == "deployment":
        if action in {"Rollback to stable deployment", "Block release and fix pipeline"}: bonus += 0.13
        elif action == "Mitigate and monitor": bonus -= 0.12
    if deployment >= 0.55 and reliability < 0.35:
        if action == "Block release and fix pipeline": bonus += 0.08
        if action == "Rollback to stable deployment": bonus -= 0.05
    if deployment >= 0.50 and reliability >= 0.35:
        if action == "Rollback to stable deployment": bonus += 0.10
        if action == "Block release and fix pipeline": bonus -= 0.03
        if action == "Mitigate and monitor": bonus -= 0.08
    if dominant == "reliability":
        if action == "Mitigate and monitor": bonus += 0.10
        if action == "Scale adjustment" and reliability >= 0.55: bonus += 0.08
        if action == "Rollback to stable deployment" and deployment < 0.40: bonus -= 0.08
    if reliability >= 0.55 and cost >= 0.15:
        if action == "Scale adjustment": bonus += 0.18
        if action == "Mitigate and monitor": bonus -= 0.08
    if reliability >= 0.55 and cost < 0.15 and deployment < 0.35:
        if action == "Mitigate and monitor": bonus += 0.10
        if action == "Scale adjustment": bonus -= 0.10
    if dominant == "cost":
        if action == "Scale adjustment": bonus += 0.22
        elif action == "Review scaling policy": bonus += 0.13
        elif action == "Mitigate and monitor": bonus -= 0.18
    if cost >= 0.45:
        if action == "Scale adjustment": bonus += 0.10
        if action == "Review scaling policy": bonus -= 0.04
        if action == "Mitigate and monitor": bonus -= 0.12
    if 0.25 <= cost < 0.45 and reliability < 0.30:
        if action == "Review scaling policy": bonus += 0.12
        if action == "Scale adjustment": bonus -= 0.04
    if dominant == "security":
        if action == "Patch or block release": bonus += 0.24
        elif action in {"Rollback to stable deployment", "Mitigate and monitor"}: bonus -= 0.15
    if security >= 0.20:
        if action == "Patch or block release": bonus += 0.16
        if action == "Rollback to stable deployment": bonus -= 0.10
        if action == "Mitigate and monitor": bonus -= 0.12
    return bonus


def choose_action_details(telemetry: Dict[str, Any], w: Tuple[float, float, float]) -> Dict[str, Any]:
    severities = severity_scores(telemetry)
    components = _action_components(telemetry)
    candidates = []
    for action, (perf, cost_eff, risk_red) in components.items():
        base = utility_score(perf, cost_eff, risk_red, w)
        bonus = _action_fit_bonus(action, severities)
        final = base + bonus
        candidates.append({"action": action, "performance_score": round(float(perf),4), "cost_efficiency_score": round(float(cost_eff),4), "risk_reduction_score": round(float(risk_red),4), "base_utility": round(float(base),4), "action_fit_bonus": round(float(bonus),4), "utility": round(float(final),4)})
    candidates.sort(key=lambda x: x["utility"], reverse=True)
    best = candidates[0]
    return {"selected_action": best["action"], "best_utility": best["utility"], "performance_score": best["performance_score"], "cost_efficiency_score": best["cost_efficiency_score"], "risk_reduction_score": best["risk_reduction_score"], "candidates": candidates[:3]}


def choose_action(telemetry: Dict[str, Any], w: Tuple[float, float, float]) -> Tuple[str, float]:
    details = choose_action_details(telemetry, w)
    return details["selected_action"], float(details["best_utility"])
