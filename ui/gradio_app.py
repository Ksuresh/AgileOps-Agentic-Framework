from __future__ import annotations

import json
import sys
from pathlib import Path

import gradio as gr

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline import run_once
from llm.deterministic_explainer import generate_explanation
from metrics.explainability import compute_xi
from pm_interface.prompt_router import route_prompt
from pm_interface.decision_formatter import format_pm_decision
from simulation.prompt_to_telemetry import build_telemetry_from_prompt_context
print(">>> LOADING NEW PHASE 2 UI <<<")

DEFAULT_THRESHOLDS = {
    "tau_consensus": 0.75,
    "delta_min": 0.15,
    "max_rar_loops": 2,
}
DEFAULT_WEIGHTS = (0.4, 0.3, 0.3)
DEFAULT_LAMBDA = 0.5

SAMPLE_PM_PROMPT = (
    "The latest release failed, users are seeing high latency and security flagged "
    "a policy violation. What should we do?"
)


def pretty_json(obj) -> str:
    return json.dumps(obj, indent=2)


def run_pm_governance(
    pm_prompt: str,
    tau_consensus: float,
    delta_min: float,
    max_rar_loops: int,
    w_perf: float,
    w_cost: float,
    w_risk: float,
    lam: float,
):
    try:
        route = route_prompt(pm_prompt)
        telemetry = build_telemetry_from_prompt_context(route)

        weights = (float(w_perf), float(w_cost), float(w_risk))
        if abs(sum(weights) - 1.0) > 1e-6:
            raise ValueError("Utility weights must sum to 1.0")

        thresholds = {
            "tau_consensus": float(tau_consensus),
            "delta_min": float(delta_min),
            "max_rar_loops": int(max_rar_loops),
        }

        result = run_once(
            telemetry=telemetry,
            thresholds=thresholds,
            lam=float(lam),
            w=weights,
        )

        payload = {
            "incident_id": "pm-interaction",
            "agents": result["agents"],
            "consensus_score": result["consensus_score"],
            "rar_triggered": result["rar_triggered"],
            "recommended_action": result["recommended_action"],
            "utility_score": result["utility_score"],
        }

        explanation = generate_explanation(payload)
        xi = compute_xi(explanation, payload)
        pm_view = format_pm_decision(pm_prompt, route, result, explanation)

        summary = {
            "recommended_action": result["recommended_action"],
            "consensus_score": result["consensus_score"],
            "rar_triggered": result["rar_triggered"],
            "rar_loops": result["rar_loops"],
            "utility_score": result["utility_score"],
        }

        return (
            pm_view,
            pretty_json(summary),
            pretty_json(route),
            pretty_json(telemetry),
            pretty_json(result["agents"]),
            pretty_json(xi),
        )

    except Exception as e:
        err = {"error": str(e)}
        msg = f"Execution failed: {e}"
        return msg, pretty_json(err), pretty_json(err), pretty_json(err), "[]", pretty_json(err)


with gr.Blocks(title="AAF Phase 2 - PM Governance Demo") as demo:
    gr.Markdown(
        """
# AgileOps Agentic Framework (AAF) - Phase 2 PM Simulation

The PM provides a natural-language request.
The system interprets the request, simulates evidence, runs multi-agent governance,
and returns a final PM-facing decision.
"""
    )

    with gr.Row():
        with gr.Column(scale=2):
            pm_prompt_input = gr.Textbox(
                label="PM Prompt",
                lines=6,
                value=SAMPLE_PM_PROMPT,
            )
            run_btn = gr.Button("Run Governance Decision")

        with gr.Column(scale=1):
            tau_input = gr.Number(label="tau_consensus", value=DEFAULT_THRESHOLDS["tau_consensus"])
            delta_input = gr.Number(label="delta_min", value=DEFAULT_THRESHOLDS["delta_min"])
            loops_input = gr.Number(label="max_rar_loops", value=DEFAULT_THRESHOLDS["max_rar_loops"], precision=0)
            lam_input = gr.Number(label="lambda", value=DEFAULT_LAMBDA)

            gr.Markdown("### Utility Weights")
            w_perf_input = gr.Number(label="w_perf", value=DEFAULT_WEIGHTS[0])
            w_cost_input = gr.Number(label="w_cost", value=DEFAULT_WEIGHTS[1])
            w_risk_input = gr.Number(label="w_risk", value=DEFAULT_WEIGHTS[2])

    pm_output = gr.Textbox(label="PM-Facing Decision", lines=18)
    summary_output = gr.Code(label="Run Summary", language="json")
    route_output = gr.Code(label="Prompt Routing", language="json")
    telemetry_output = gr.Code(label="Simulated Telemetry", language="json")
    agents_output = gr.Code(label="Agent Outputs", language="json")
    xi_output = gr.Code(label="Explainability (XI)", language="json")

    run_btn.click(
        fn=run_pm_governance,
        inputs=[
            pm_prompt_input,
            tau_input,
            delta_input,
            loops_input,
            w_perf_input,
            w_cost_input,
            w_risk_input,
            lam_input,
        ],
        outputs=[
            pm_output,
            summary_output,
            route_output,
            telemetry_output,
            agents_output,
            xi_output,
        ],
    )


if __name__ == "__main__":
    demo.launch()
