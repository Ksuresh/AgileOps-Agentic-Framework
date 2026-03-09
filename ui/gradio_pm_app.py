from __future__ import annotations

import json
import sys
from pathlib import Path

import gradio as gr
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline import run_once
from llm.deterministic_explainer import generate_explanation
from metrics.explainability import compute_xi
from pm_interface.prompt_router import route_prompt
from pm_interface.decision_formatter import format_pm_decision
from simulation.prompt_to_telemetry import build_telemetry_from_prompt_context


DEFAULT_THRESHOLDS = {
    "tau_consensus": 0.75,
    "delta_min": 0.15,
    "max_rar_loops": 2,
}

DEFAULT_WEIGHTS = (0.4, 0.3, 0.3)
DEFAULT_LAMBDA = 0.5

PROMPT_LIBRARY_PATH = REPO_ROOT / "prompts" / "pm_prompt_library.yaml"


def pretty_json(obj) -> str:
    return json.dumps(obj, indent=2)


def load_prompt_library() -> list[dict]:
    if not PROMPT_LIBRARY_PATH.exists():
        return []
    data = yaml.safe_load(PROMPT_LIBRARY_PATH.read_text(encoding="utf-8")) or {}
    prompts = data.get("prompts", [])
    return prompts if isinstance(prompts, list) else []


PROMPT_LIBRARY = load_prompt_library()


def get_categories() -> list[str]:
    cats = sorted({p.get("category", "uncategorized") for p in PROMPT_LIBRARY})
    return ["all"] + cats


def get_prompt_choices(category: str) -> list[str]:
    items = PROMPT_LIBRARY
    if category and category != "all":
        items = [p for p in PROMPT_LIBRARY if p.get("category") == category]
    return [f"{p.get('id')} | {p.get('title')}" for p in items]


def get_prompt_by_choice(choice: str) -> dict | None:
    if not choice:
        return None
    prefix = choice.split("|", 1)[0].strip()
    for p in PROMPT_LIBRARY:
        if p.get("id") == prefix:
            return p
    return None


def on_category_change(category: str):
    choices = get_prompt_choices(category)
    first = choices[0] if choices else None
    prompt_text = ""
    metadata = {}
    if first:
        item = get_prompt_by_choice(first)
        if item:
            prompt_text = item.get("prompt", "")
            metadata = {
                "id": item.get("id"),
                "title": item.get("title"),
                "category": item.get("category"),
                "priority": item.get("priority"),
                "expected_domains": item.get("expected_domains", []),
                "expected_action": item.get("expected_action"),
                "expected_business_focus": item.get("expected_business_focus"),
                "notes": item.get("notes"),
            }

    return (
        gr.update(choices=choices, value=first),
        prompt_text,
        pretty_json(metadata),
    )


def on_prompt_choice_change(choice: str):
    item = get_prompt_by_choice(choice)
    if not item:
        return "", pretty_json({})
    metadata = {
        "id": item.get("id"),
        "title": item.get("title"),
        "category": item.get("category"),
        "priority": item.get("priority"),
        "expected_domains": item.get("expected_domains", []),
        "expected_action": item.get("expected_action"),
        "expected_business_focus": item.get("expected_business_focus"),
        "notes": item.get("notes"),
    }
    return item.get("prompt", ""), pretty_json(metadata)


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
        weights = (float(w_perf), float(w_cost), float(w_risk))
        if abs(sum(weights) - 1.0) > 1e-6:
            raise ValueError("Utility weights must sum to 1.0")

        route = route_prompt(pm_prompt)
        telemetry = build_telemetry_from_prompt_context(route)

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


INITIAL_CATEGORY = get_categories()[0] if get_categories() else "all"
INITIAL_CHOICES = get_prompt_choices(INITIAL_CATEGORY)
INITIAL_CHOICE = INITIAL_CHOICES[0] if INITIAL_CHOICES else None
INITIAL_ITEM = get_prompt_by_choice(INITIAL_CHOICE) if INITIAL_CHOICE else None
INITIAL_PROMPT = INITIAL_ITEM.get("prompt", "") if INITIAL_ITEM else ""
INITIAL_METADATA = {
    "id": INITIAL_ITEM.get("id") if INITIAL_ITEM else None,
    "title": INITIAL_ITEM.get("title") if INITIAL_ITEM else None,
    "category": INITIAL_ITEM.get("category") if INITIAL_ITEM else None,
    "priority": INITIAL_ITEM.get("priority") if INITIAL_ITEM else None,
    "expected_domains": INITIAL_ITEM.get("expected_domains", []) if INITIAL_ITEM else [],
    "expected_action": INITIAL_ITEM.get("expected_action") if INITIAL_ITEM else None,
    "expected_business_focus": INITIAL_ITEM.get("expected_business_focus") if INITIAL_ITEM else None,
    "notes": INITIAL_ITEM.get("notes") if INITIAL_ITEM else None,
}


with gr.Blocks(title="AAF Phase 2 – PM Governance Demo") as demo:
    gr.Markdown(
        """
# AgileOps Agentic Framework – PM Prompt Mode

The PM selects or edits a natural-language request.
The system interprets it, simulates telemetry, runs multi-agent governance,
and returns a PM-facing governance decision.
"""
    )

    with gr.Row():
        with gr.Column(scale=2):
            category_dropdown = gr.Dropdown(
                label="Prompt Category",
                choices=get_categories(),
                value=INITIAL_CATEGORY,
            )

            prompt_dropdown = gr.Dropdown(
                label="Prompt Scenario",
                choices=INITIAL_CHOICES,
                value=INITIAL_CHOICE,
            )

            pm_prompt_input = gr.Textbox(
                label="PM Prompt",
                lines=6,
                value=INITIAL_PROMPT,
            )

            run_btn = gr.Button("Run Governance Decision")

        with gr.Column(scale=1):
            prompt_metadata_output = gr.Code(
                label="Selected Prompt Metadata",
                language="json",
                value=pretty_json(INITIAL_METADATA),
            )

            tau_input = gr.Number(label="tau_consensus", value=DEFAULT_THRESHOLDS["tau_consensus"])
            delta_input = gr.Number(label="delta_min", value=DEFAULT_THRESHOLDS["delta_min"])
            loops_input = gr.Number(
                label="max_rar_loops",
                value=DEFAULT_THRESHOLDS["max_rar_loops"],
                precision=0,
            )
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

    category_dropdown.change(
        fn=on_category_change,
        inputs=[category_dropdown],
        outputs=[prompt_dropdown, pm_prompt_input, prompt_metadata_output],
    )

    prompt_dropdown.change(
        fn=on_prompt_choice_change,
        inputs=[prompt_dropdown],
        outputs=[pm_prompt_input, prompt_metadata_output],
    )

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
    demo.launch(server_port=7861)
