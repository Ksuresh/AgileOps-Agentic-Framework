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


with gr.Blocks(title="AAF Telemetry Demo") as demo:

    gr.Markdown(
        """
# AgileOps Agentic Framework (AAF) – Telemetry Mode

Paste telemetry JSON and run the governance pipeline.
"""
    )

    telemetry_input = gr.Code(
        label="Telemetry JSON",
        language="json",
        value=pretty_json(SAMPLE_TELEMETRY),
        lines=20,
    )

    run_btn = gr.Button("Run AAF")

    summary_output = gr.Code(label="Run Summary", language="json")
    agents_output = gr.Code(label="Agent Outputs", language="json")
    explanation_output = gr.Textbox(label="Explanation", lines=10)
    xi_output = gr.Code(label="Explainability (XI)", language="json")

    run_btn.click(
        fn=run_aaf,
        inputs=[
            telemetry_input,
            gr.Number(value=DEFAULT_THRESHOLDS["tau_consensus"]),
            gr.Number(value=DEFAULT_THRESHOLDS["delta_min"]),
            gr.Number(value=DEFAULT_THRESHOLDS["max_rar_loops"]),
            gr.Number(value=DEFAULT_WEIGHTS[0]),
            gr.Number(value=DEFAULT_WEIGHTS[1]),
            gr.Number(value=DEFAULT_WEIGHTS[2]),
            gr.Number(value=DEFAULT_LAMBDA),
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
