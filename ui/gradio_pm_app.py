from __future__ import annotations

import json
from typing import Any, Dict, Tuple

import gradio as gr

from pipeline import run_pipeline


DEFAULT_THRESHOLDS = {
    "tau_consensus": 0.65,
    "delta_min": 0.05,
    "max_rar_loops": 2,
}

DEFAULT_UTILITY_WEIGHTS = (0.4, 0.3, 0.3)


EXAMPLE_PROMPTS = [
    "A release went out this morning and customer checkout latency increased. Should we rollback or monitor?",
    "Cloud spend increased after autoscaling, but the service appears stable. What should the PM do?",
    "A critical CVE was reported for a container image included in the planned release. Can the release proceed?",
    "The service shows high CPU saturation and rising latency during peak traffic. Should we scale?",
    "The deployment pipeline failed a release gate because compliance evidence is missing. What is the recommended governance action?",
]


def _prompt_to_basic_telemetry(prompt: str) -> Dict[str, Any]:
    p = prompt.lower()

    telemetry = {
        "deploy": {
            "restart_loops": 0,
            "config_drift": False,
            "pipeline_failed": False,
            "rollback_marker": False,
            "artifact_mismatch": False,
        },
        "sre": {
            "p95_latency_ms": 180.0,
            "error_rate_pct": 0.5,
            "saturation_pct": 55.0,
            "availability_pct": 99.9,
        },
        "finops": {
            "cost_spike_pct": 0.0,
            "hpa_scale_to": 4,
            "cpu_request_increase_pct": 0.0,
            "memory_request_increase_pct": 0.0,
        },
        "sec": {
            "critical_cves": 0,
            "policy_violation": False,
            "iam_drift": False,
            "compliance_gap": False,
        },
    }

    if any(k in p for k in ["deployment", "release", "rollback", "pipeline", "build", "artifact"]):
        telemetry["deploy"]["pipeline_failed"] = "pipeline" in p or "build" in p
        telemetry["deploy"]["rollback_marker"] = "rollback" in p or "release" in p
        telemetry["deploy"]["artifact_mismatch"] = "artifact" in p or "image" in p
        telemetry["deploy"]["restart_loops"] = 12 if "restart" in p or "crash" in p else 8

    if any(k in p for k in ["config", "configuration", "drift"]):
        telemetry["deploy"]["config_drift"] = True

    if any(k in p for k in ["latency", "slow", "timeout", "performance"]):
        telemetry["sre"]["p95_latency_ms"] = 650.0
        telemetry["sre"]["availability_pct"] = 98.4

    if any(k in p for k in ["error", "5xx", "failure rate", "failed requests"]):
        telemetry["sre"]["error_rate_pct"] = 10.0
        telemetry["sre"]["availability_pct"] = 98.0

    if any(k in p for k in ["cpu", "memory", "saturation", "capacity"]):
        telemetry["sre"]["saturation_pct"] = 90.0
        telemetry["sre"]["p95_latency_ms"] = max(telemetry["sre"]["p95_latency_ms"], 520.0)

    if any(k in p for k in ["cost", "spend", "budget", "finops", "cloud bill"]):
        telemetry["finops"]["cost_spike_pct"] = 32.0
        telemetry["finops"]["hpa_scale_to"] = 12

    if any(k in p for k in ["scale", "autoscale", "autoscaling", "hpa", "replica"]):
        telemetry["finops"]["hpa_scale_to"] = 14
        telemetry["finops"]["cost_spike_pct"] = max(telemetry["finops"]["cost_spike_pct"], 24.0)

    if any(k in p for k in ["over provision", "over-provision", "unused capacity"]):
        telemetry["finops"]["cpu_request_increase_pct"] = 60.0
        telemetry["finops"]["memory_request_increase_pct"] = 45.0
        telemetry["finops"]["cost_spike_pct"] = max(telemetry["finops"]["cost_spike_pct"], 28.0)

    if any(k in p for k in ["security", "vulnerability", "cve", "critical cve"]):
        telemetry["sec"]["critical_cves"] = 2

    if any(k in p for k in ["policy", "opa", "gatekeeper", "compliance"]):
        telemetry["sec"]["policy_violation"] = True
        telemetry["sec"]["compliance_gap"] = True

    if any(k in p for k in ["iam", "permission", "access drift"]):
        telemetry["sec"]["iam_drift"] = True

    return telemetry


def _build_scenario(
    prompt: str,
    tau_consensus: float,
    delta_min: float,
    w_perf: float,
    w_cost: float,
    w_risk: float,
) -> Dict[str, Any]:
    telemetry = None

    try:
        from pm_interface.prompt_router import route_prompt
        from simulation.prompt_to_telemetry import prompt_to_telemetry

        routed = route_prompt(prompt)
        telemetry = prompt_to_telemetry(routed)
    except Exception:
        telemetry = _prompt_to_basic_telemetry(prompt)

    if not isinstance(telemetry, dict):
        telemetry = _prompt_to_basic_telemetry(prompt)

    return {
        "scenario_id": "PM-UI-001",
        "incident_id": "PM-UI-001",
        "category": "pm_prompt_ui",
        "scenario_type": "pm_prompt_ui",
        "prompt": prompt,
        "telemetry": telemetry,
        "ground_truth": {
            "primary_domain": None,
            "secondary_domains": [],
            "root_cause": "pm_prompt_ui",
            "recommended_action": None,
            "expected_action": None,
        },
        "thresholds": {
            "tau_consensus": float(tau_consensus),
            "delta_min": float(delta_min),
            "max_rar_loops": 2,
        },
        "utility_weights": (
            float(w_perf),
            float(w_cost),
            float(w_risk),
        ),
        "lam": 0.5,
    }


def _safe_json(data: Any) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False)


def run_pm_prompt(
    prompt: str,
    tau_consensus: float,
    delta_min: float,
    w_perf: float,
    w_cost: float,
    w_risk: float,
) -> Tuple[str, str, str, str, str]:
    if not prompt or not prompt.strip():
        return (
            "Please enter a PM governance prompt.",
            "{}",
            "{}",
            "{}",
            "{}",
        )

    scenario = _build_scenario(
        prompt=prompt.strip(),
        tau_consensus=tau_consensus,
        delta_min=delta_min,
        w_perf=w_perf,
        w_cost=w_cost,
        w_risk=w_risk,
    )

    result = run_pipeline(scenario, mode="aaf_full")
    row = result.__dict__.copy()

    decision_summary = f"""
## PM Governance Decision

**Predicted Primary Domain:** {row.get("predicted_primary_domain")}

**Recommended Action:** {(row.get("utility") or {}).get("selected_action")}

**Consensus Score:** {row.get("consensus_score")}

**RAR Triggered:** {(row.get("rar") or {}).get("triggered")}

**RAR Accepted:** {(row.get("rar") or {}).get("accepted")}

**Composite Utility:** {(row.get("utility") or {}).get("best_utility")}

**Explainability Index:** {(row.get("explainability") or {}).get("xi")}

---

{row.get("explanation", "")}
""".strip()

    agents = row.get("agents", [])
    utility = row.get("utility", {})
    explainability = row.get("explainability", {})
    telemetry = scenario.get("telemetry", {})

    return (
        decision_summary,
        _safe_json(telemetry),
        _safe_json(agents),
        _safe_json(utility),
        _safe_json(explainability),
    )


def build_app() -> gr.Blocks:
    with gr.Blocks(title="AgileOps Agentic Framework - PM Governance UI") as demo:
        gr.Markdown(
            """
# AgileOps Agentic Framework - PM Governance UI

Enter a Project Manager governance prompt. The framework routes the prompt into operational telemetry, runs DevOps/SRE/FinOps/DevSecOps agents, checks consensus, applies RAR when needed, selects a utility-based action, and returns a PM-readable explanation.
"""
        )

        with gr.Row():
            prompt = gr.Textbox(
                label="PM Governance Prompt",
                lines=5,
                value=EXAMPLE_PROMPTS[0],
            )

        with gr.Row():
            example = gr.Dropdown(
                choices=EXAMPLE_PROMPTS,
                value=EXAMPLE_PROMPTS[0],
                label="Example prompts",
            )

        example.change(fn=lambda x: x, inputs=example, outputs=prompt)

        with gr.Accordion("Advanced Settings", open=False):
            with gr.Row():
                tau_consensus = gr.Slider(
                    minimum=0.40,
                    maximum=0.90,
                    value=DEFAULT_THRESHOLDS["tau_consensus"],
                    step=0.01,
                    label="Consensus Threshold",
                )
                delta_min = gr.Slider(
                    minimum=0.00,
                    maximum=0.25,
                    value=DEFAULT_THRESHOLDS["delta_min"],
                    step=0.01,
                    label="RAR Minimum Improvement",
                )

            with gr.Row():
                w_perf = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=DEFAULT_UTILITY_WEIGHTS[0],
                    step=0.05,
                    label="Utility Weight: Performance",
                )
                w_cost = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=DEFAULT_UTILITY_WEIGHTS[1],
                    step=0.05,
                    label="Utility Weight: Cost Efficiency",
                )
                w_risk = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=DEFAULT_UTILITY_WEIGHTS[2],
                    step=0.05,
                    label="Utility Weight: Risk Reduction",
                )

        run_button = gr.Button("Run Governance Analysis", variant="primary")

        decision = gr.Markdown(label="PM Decision Summary")

        with gr.Tab("Telemetry"):
            telemetry_out = gr.Code(label="Telemetry", language="json")

        with gr.Tab("Agent Outputs"):
            agents_out = gr.Code(label="Agents", language="json")

        with gr.Tab("Utility"):
            utility_out = gr.Code(label="Utility", language="json")

        with gr.Tab("Explainability"):
            explainability_out = gr.Code(label="Explainability", language="json")

        run_button.click(
            fn=run_pm_prompt,
            inputs=[
                prompt,
                tau_consensus,
                delta_min,
                w_perf,
                w_cost,
                w_risk,
            ],
            outputs=[
                decision,
                telemetry_out,
                agents_out,
                utility_out,
                explainability_out,
            ],
        )

    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch()
