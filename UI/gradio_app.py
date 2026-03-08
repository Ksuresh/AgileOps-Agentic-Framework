from __future__ import annotations

import json
import sys
from pathlib import Path

import gradio as gr

# Ensure repo root is on sys.path when running as a script
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline import run_once
from llm.deterministic_explainer import generate_explanation
from metrics.explainability import compute_xi


DEFAULT_THRESHOLDS = {
    "tau_consensus": 0.75,
    "delta_min": 0.15,
    "max_rar_loops": 2,
}

DEFAULT_WEIGHTS = (0.4, 0.3, 0.3)
DEFAULT_LAMBDA = 0.5


SAMPLE_TELEMETRY = {
    "deploy": {
        "restart_loops": 18,
        "config_drift": True,
        "pipeline_failed": True,
    },
    "sre": {
        "p95_latency_ms": 900.0,
        "error_rate_pct": 14.0,
        "saturation_pct": 92.0,
    },
    "finops": {
        "cost_spike_pct": 22.0,
        "hpa_scale_to": 11,
    },
    "sec": {
        "critical_cves": 1,
        "policy_violation": True,
        "iam_drift": False,
    },
}


def pretty_json(obj) -> str:
    return json.dumps(obj, indent=2)


def run_aaf(
    telemetry_text: str,
    tau_consensus: float,
    delta_min: float,
    max_rar_loops: int,
    w_perf: float,
    w_cost: float,
    w_risk: float,
    lam: float,
):
    try:
        telemetry = json.loads(telemetry_text)

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
            "incident_id": "interactive-demo",
            "agents": result["agents"],
            "consensus_score": result["consensus_score"],
            "rar_triggered": result["rar_triggered"],
            "recommended_action": result["recommended_action"],
            "utility_score": result["utility_score"],
        }

        explanation = generate_explanation(payload)
        xi = compute_xi(explanation, payload)

        summary = {
            "consensus_score": result["consensus_score"],
            "rar_triggered": result["rar_triggered"],
            "rar_loops": result["rar_loops"],
            "recommended_action": result["recommended_action"],
            "utility_score": result["utility_score"],
        }

        return (
            pretty_json(summary),
            pretty_json(result["agents"]),
            explanation,
            pretty_json(xi),
        )

    except Exception as e:
        err = {"error": str(e)}
        return pretty_json(err), "[]", "Execution failed.", pretty_json(err)


def load_sample():
    return pretty_json(SAMPLE_TELEMETRY)


with gr.Blocks(title="AgileOps Agentic Framework (AAF) Demo") as demo:
    gr.Markdown(
        """
# AgileOps Agentic Framework (AAF) Demo

Paste telemetry JSON and run the synthetic AAF pipeline.

This demo shows:

- DevOps / SRE / FinOps / DevSecOps agent outputs
- Consensus score
- RAR trigger behavior
- Utility-based action recommendation
- Deterministic explanation
- Explainability Index (XI)
"""
    )

    with gr.Row():
        with gr.Column(scale=2):
            telemetry_input = gr.Code(
                label="Telemetry JSON",
                language="json",
                value=pretty_json(SAMPLE_TELEMETRY),
                lines=24,
            )

            with gr.Row():
                load_sample_btn = gr.Button("Load Sample Telemetry")
                run_btn = gr.Button("Run AAF")

        with gr.Column(scale=1):
            tau_input = gr.Number(
                label="tau_consensus",
                value=DEFAULT_THRESHOLDS["tau_consensus"],
            )
            delta_input = gr.Number(
                label="delta_min",
                value=DEFAULT_THRESHOLDS["delta_min"],
            )
            loops_input = gr.Number(
                label="max_rar_loops",
                value=DEFAULT_THRESHOLDS["max_rar_loops"],
                precision=0,
            )
            lam_input = gr.Number(
                label="lambda",
                value=DEFAULT_LAMBDA,
            )

            gr.Markdown("### Utility Weights")
            w_perf_input = gr.Number(label="w_perf", value=DEFAULT_WEIGHTS[0])
            w_cost_input = gr.Number(label="w_cost", value=DEFAULT_WEIGHTS[1])
            w_risk_input = gr.Number(label="w_risk", value=DEFAULT_WEIGHTS[2])

    with gr.Row():
        summary_output = gr.Code(label="Run Summary", language="json")
        xi_output = gr.Code(label="Explainability (XI)", language="json")

    agents_output = gr.Code(label="Agent Outputs", language="json", lines=22)
    explanation_output = gr.Textbox(label="Deterministic Explanation", lines=14)

    load_sample_btn.click(
        fn=load_sample,
        outputs=[telemetry_input],
    )

    run_btn.click(
        fn=run_aaf,
        inputs=[
            telemetry_input,
            tau_input,
            delta_input,
            loops_input,
            w_perf_input,
            w_cost_input,
            w_risk_input,
            lam_input,
        ],
        outputs=[
            summary_output,
            agents_output,
            explanation_output,
            xi_output,
        ],
    )


if __name__ == "__main__":
    demo.launch()
