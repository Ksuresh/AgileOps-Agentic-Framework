from __future__ import annotations
from typing import Dict, Any, Tuple, Literal
from dataclasses import dataclass
import time
from agents.devops import DevOpsAgent
from agents.sre import SREAgent
from agents.finops import FinOpsAgent
from agents.devsecops import DevSecOpsAgent
from orchestrator.consensus import consensus_score
from orchestrator.rar import re_ground_telemetry
from orchestrator.utility import choose_action

from llm.deterministic_explainer import generate_explanation
from metrics.explainability import compute_xi

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


# ---------------------------------------------------------------------------
# Paper-friendly wrapper (modes, timings, XI)

Mode = Literal["aaf_full", "aaf_no_consensus", "aaf_no_rar", "aaf_no_utility"]


@dataclass
class PipelineResult:
    scenario_id: str
    ground_truth: Dict[str, Any]
    mode: str
    predicted_primary_domain: str | None
    agents: list
    consensus_score: float
    rar: Dict[str, Any]
    utility: Dict[str, Any]
    explanation: str
    explainability: Dict[str, float]
    timings: Dict[str, float]


def run_pipeline(scenario: Dict[str, Any], mode: Mode = "aaf_full") -> PipelineResult:
    """Run the pipeline in a way that can directly back the paper tables.

    - Adds stage timings (ms): T-IN, AG-INF, CN-CHK, RAR, LLM-XP, OUT-GEN, TOTAL
    - Adds deterministic explanation + XI for reproducibility without APIs.
    - Adds ablation modes.
    """
    t0 = time.perf_counter()
    timings: Dict[str, float] = {}

    telemetry = scenario.get("telemetry", {})
    thresholds = scenario.get("thresholds", {"tau_consensus": 0.75, "delta_min": 0.15, "max_rar_loops": 2})
    lam = float(scenario.get("lam", 0.6))
    w = tuple(scenario.get("utility_weights", (0.4, 0.3, 0.3)))  # type: ignore

    # T-IN
    t_in = time.perf_counter()
    _ = dict(telemetry)
    timings["T-IN"] = (time.perf_counter() - t_in) * 1000.0

    # AG-INF
    t_ag = time.perf_counter()
    outputs = [a.infer(telemetry) for a in AGENTS]
    timings["AG-INF"] = (time.perf_counter() - t_ag) * 1000.0

    # CN-CHK
    t_cn = time.perf_counter()
    claims = [o.claim for o in outputs]
    confs = [float(o.confidence) for o in outputs]
    if mode == "aaf_no_consensus":
        s = 1.0
    else:
        s, _ = consensus_score(claims, confs, lam=lam)
    timings["CN-CHK"] = (time.perf_counter() - t_cn) * 1000.0

    # RAR
    rar_info = {"triggered": False, "accepted": True, "before": float(s), "after": float(s)}
    timings["RAR"] = 0.0
    tau = float(thresholds.get("tau_consensus", 0.75))
    delta_min = float(thresholds.get("delta_min", 0.15))
    max_loops = int(thresholds.get("max_rar_loops", 2))

    rar_triggered = False
    loops = 0
    t_cur = telemetry

    if mode != "aaf_no_rar":
        while s < tau and loops < max_loops:
            rar_triggered = True
            loops += 1
            t_rar = time.perf_counter()
            t_cur, s_new, accepted = re_ground_telemetry(
                telemetry=t_cur,
                tau=tau,
                delta_min=delta_min,
                lam=lam,
            )
            outputs = [a.infer(t_cur) for a in AGENTS]
            claims = [o.claim for o in outputs]
            confs = [float(o.confidence) for o in outputs]
            timings["RAR"] += (time.perf_counter() - t_rar) * 1000.0

            s = s_new
            if not accepted:
                rar_info["accepted"] = False
                break

    rar_info["triggered"] = bool(rar_triggered)
    rar_info["after"] = float(s)
    rar_info["loops"] = int(loops)

    # Utility
    t_ut = time.perf_counter()
    if mode == "aaf_no_utility":
        action, util = ("defer", 0.0)
        candidates = []
    else:
        action, util = choose_action(t_cur, w)
        candidates = []
    timings["UTL"] = (time.perf_counter() - t_ut) * 1000.0

    # Explanation
    t_xp = time.perf_counter()
    payload = {
        "incident_id": scenario.get("scenario_id"),
        "agents": [
            {
                "agent_type": o.agent_type,
                "claim": o.claim,
                "confidence": float(o.confidence),
                "evidence": list(o.evidence) if getattr(o, "evidence", None) else [],
            }
            for o in outputs
        ],
        "consensus_score": float(s),
        "rar_triggered": bool(rar_triggered),
        "recommended_action": action,
        "utility_score": float(util),
    }
    explanation = generate_explanation(payload)
    timings["LLM-XP"] = (time.perf_counter() - t_xp) * 1000.0

    t_out = time.perf_counter()
    explainability = compute_xi(explanation, payload)
    timings["OUT-GEN"] = (time.perf_counter() - t_out) * 1000.0

    timings["TOTAL"] = (time.perf_counter() - t0) * 1000.0

    pred = _predict_primary_domain(outputs)

    return PipelineResult(
        scenario_id=str(scenario.get("scenario_id")),
        ground_truth=scenario.get("ground_truth", {}),
        mode=mode,
        predicted_primary_domain=pred,
        agents=[o.__dict__ for o in outputs],
        consensus_score=float(s),
        rar=rar_info,
        utility={"selected_action": action, "best_utility": float(util), "candidates": candidates},
        explanation=explanation,
        explainability=explainability,
        timings=timings,
    )


def _predict_primary_domain(outputs: list) -> str | None:
    """Pick the agent with highest confidence among non-trivial claims."""
    filtered = []
    for o in outputs:
        c = (o.claim or "").lower()
        if c.startswith("no ") or "no issue" in c or "no violation" in c:
            continue
        filtered.append(o)
    if not filtered:
        filtered = outputs
    best = max(filtered, key=lambda x: float(x.confidence))
    return getattr(best, "agent_type", None)
