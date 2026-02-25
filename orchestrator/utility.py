from __future__ import annotations
from typing import Dict, Any, Tuple

def utility_score(perf_gain: float, cost_delta: float, risk: float, w: Tuple[float,float,float]) -> float:
    w_perf, w_cost, w_risk = w
    return w_perf*perf_gain - w_cost*cost_delta - w_risk*risk

def choose_action(telemetry: Dict[str, Any], w: Tuple[float,float,float]) -> Tuple[str, float]:
    actions = {
        "Rollback to stable deployment": (0.80, 0.20, 0.25),
        "Mitigate and monitor":          (0.55, 0.10, 0.20),
        "Scale adjustment":             (0.60, 0.35, 0.15),
        "Patch or block release":       (0.50, 0.15, 0.60),
        "No action (observe)":          (0.20, 0.00, 0.05),
    }
    best_a, best_u = None, -1e9
    for a,(p,c,r) in actions.items():
        u = utility_score(p,c,r,w)
        if u > best_u:
            best_u = u
            best_a = a
    return best_a, float(best_u)
