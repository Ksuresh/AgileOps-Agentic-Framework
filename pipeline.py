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


def _run_agents(telemetry: Dict[str, Any]):
    outputs = [a.infer(telemetry) for a in AGENTS]
    claims = [o.claim for o in outputs]
    confs = [float(o.confidence) for o in outputs]
    return outputs, claims, confs


def run_once(
    telemetry: Dict[str, Any],
    thresholds: Dict[str, Any],
    lam: float,
    w: Tuple[float, float, float],
) -> Dict[str, Any]:
    """
    Core reproducibility execution path.

    Steps:
    1. Run all domain agents
    2. Compute consensus
    3. Trigger RAR if consensus is below threshold
    4. Re-run agents if RAR accepted
    5. Select recommended action from telemetry-aware utility
    """
    outputs, claims, confs = _run_agents(telemetry)
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

        t_updated, s_after, accepted = re_ground_telemetry(
            telemetry=t,
            tau=tau,
            delta_min=delta_min,
            lam=lam,
        )

        t = t_updated
        outputs, claims, confs = _run_agents(t)
        s_recomputed, _ = consensus_score(claims, confs, lam=lam)
        s = float(s_recomputed)

        if not accepted:
            break

        if s_after < tau:
            continue

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
    """
    Paper-oriented pipeline runner.

    Adds:
    - stage timings
    - deterministic explanation
    - explainability index
    - ablation modes
    """
    t0 = time.perf_counter()
    timings: Dict[str, float] = {}

    telemetry = scenario.get("telemetry", {})
    thresholds = scenario.get(
        "thresholds",
        {"tau_consensus": 0.75, "delta_min": 0.15, "max_rar_loops": 2},
    )
    lam = float(scenario.get("lam", 0.5))
    w = tuple(scenario.get("utility_weights", (0.4, 0.3, 0.3)))  # type: ignore

    # T-IN
    t_in = time.perf_counter()
    _ = dict(telemetry)
    timings["T-IN"] = (time.perf_counter() - t_in) * 1000.0

    # AG-INF
    t_ag = time.perf_counter()
    outputs, claims, confs = _run_agents(telemetry)
    timings["AG-INF"] = (time.perf_counter() - t_ag) * 1000.0

    # CN-CHK
    t_cn = time.perf_counter()
    if mode == "aaf_no_consensus":
        s = 1.0
    else:
        s, _ = consensus_score(claims, confs, lam=lam)
    timings["CN-CHK"] = (time.perf_counter() - t_cn) * 1000.0

    # RAR
    rar_info = {
        "triggered": False,
        "accepted": True,
        "before": float(s),
        "after": float(s),
        "loops": 0,
    }
    timings["RAR"] = 0.0

    tau = float(thresholds.get("tau_consensus", 0.75))
    delta_min = float(thresholds.get("delta_min", 0.15))
    max_loops = int(thresholds.get("max_rar_loops", 2))

    t_cur = telemetry

    if mode != "aaf_no_rar":
        loops = 0
        while s < tau and loops < max_loops:
            loops += 1
            rar_info["triggered"] = True

            t_rar = time.perf_counter()
            t_updated, s_after, accepted = re_ground_telemetry(
                telemetry=t_cur,
                tau=tau,
                delta_min=delta_min,
                lam=lam,
            )
            timings["RAR"] += (time.perf_counter() - t_rar) * 1000.0

            t_cur = t_updated
            outputs, claims, confs = _run_agents(t_cur)
            s, _ = consensus_score(claims, confs, lam=lam)

            rar_info["accepted"] = bool(accepted)
            rar_info["after"] = float(s)
            rar_info["loops"] = loops

            if not accepted:
                break

            if s_after < tau:
                continue

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
        "incident_id": scenario.get("scenario_id", scenario.get("incident_id")),
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
        "rar_triggered": bool(rar_info["triggered"]),
        "recommended_action": action,
        "utility_score": float(util),
    }
    explanation = generate_explanation(payload)
    timings["LLM-XP"] = (time.perf_counter() - t_xp) * 1000.0

    # Output generation / explainability
    t_out = time.perf_counter()
    explainability = compute_xi(explanation, payload)
    timings["OUT-GEN"] = (time.perf_counter() - t_out) * 1000.0

    timings["TOTAL"] = (time.perf_counter() - t0) * 1000.0

    pred = _predict_primary_domain(outputs)

    return PipelineResult(
        scenario_id=str(scenario.get("scenario_id", scenario.get("incident_id", "unknown"))),
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
    """
    Pick the agent with highest confidence among non-trivial claims.
    """
    filtered = []
    for o in outputs:
        claim = (o.claim or "").lower()
        if claim.startswith("no ") or "no issue" in claim or "no violation" in claim:
            continue
        filtered.append(o)

    if not filtered:
        filtered = outputs

    best = max(filtered, key=lambda x: float(x.confidence))
    return getattr(best, "agent_type", None)
