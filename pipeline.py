from __future__ import annotations
from typing import Dict, Any, Tuple
from agents.devops import DevOpsAgent
from agents.sre import SREAgent
from agents.finops import FinOpsAgent
from agents.devsecops import DevSecOpsAgent
from orchestrator.consensus import consensus_score
from orchestrator.rar import re_ground
from orchestrator.utility import choose_action

AGENTS = [DevOpsAgent(), SREAgent(), FinOpsAgent(), DevSecOpsAgent()]

def run_once(telemetry: Dict[str, Any], thresholds: Dict[str, Any], lam: float, w: Tuple[float,float,float]) -> Dict[str, Any]:
    outputs = [a.infer(telemetry) for a in AGENTS]
    claims = [o.claim for o in outputs]
    confs = [float(o.confidence) for o in outputs]
    s, _ = consensus_score(claims, confs, lam=lam)

    tau = float(thresholds["tau_consensus"])
    delta_min = float(thresholds["delta_min"])
    max_loops = int(thresholds.get("max_rar_loops", 2))

    rar_triggered = False
    loops = 0
    t = telemetry

    while s < tau and loops < max_loops:
        rar_triggered = True
        loops += 1
        t = re_ground(t)
        outputs = [a.infer(t) for a in AGENTS]
        claims = [o.claim for o in outputs]
        confs = [float(o.confidence) for o in outputs]
        s_new, _ = consensus_score(claims, confs, lam=lam)
        if (s_new - s) >= delta_min:
            s = s_new
        else:
            s = s_new
            break

    action, util = choose_action(t, w)

    return {
        "agents": [o.__dict__ for o in outputs],
        "consensus_score": float(s),
        "rar_triggered": bool(rar_triggered),
        "rar_loops": int(loops),
        "recommended_action": action,
        "utility_score": float(util),
    }
