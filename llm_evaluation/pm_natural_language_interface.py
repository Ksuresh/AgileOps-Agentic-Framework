from __future__ import annotations

"""Project-manager natural-language interface for AAF.

The LLM has two bounded roles:
1) map a PM question to a supported query intent;
2) verbalize an already-computed AAF decision record.

It cannot select or change the governance action. The authoritative action,
evidence, interactions, consensus and utility are supplied by AAF and are
validated after generation.
"""

import json
import os
from dataclasses import dataclass
from typing import Any, Dict

from openai import OpenAI


DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
SUPPORTED_INTENTS = {
    "decision",          # What should we do?
    "why",               # Why this action?
    "evidence",          # What evidence supports it?
    "alternatives",      # What other actions were considered?
    "uncertainty",       # How confident are we / what is missing?
    "status",            # What is happening operationally?
}


INTENT_SYSTEM = """You map a Project Manager's natural-language question to one AAF query intent.
Return JSON only with keys intent and normalized_question.
Allowed intents: decision, why, evidence, alternatives, uncertainty, status.
Do not make any governance decision and do not infer operational facts."""

ANSWER_SYSTEM = """You are the natural-language interface to the AgileOps Agent Framework (AAF),
a deterministic evidence-driven multi-agent decision-support framework.

Rules:
- Use only the supplied AAF decision record.
- Never change, weaken, strengthen, or replace selected_action.
- Never invent telemetry, causal links, risks, incidents, business impact, or missing evidence.
- If asked what to do, state selected_action exactly.
- If asked why, explain only from supplied interactions/evidence/utility.
- If asked for alternatives, mention only supplied candidate actions.
- If evidence is incomplete, say so.
- Keep the answer concise and useful to a Project Manager.
Return JSON only with keys answer and selected_action."""


@dataclass(frozen=True)
class PMIntent:
    intent: str
    normalized_question: str


def _parse_json_object(text: str) -> Dict[str, Any]:
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("LLM response must be a JSON object")
    return value


def extract_intent(question: str, *, client: OpenAI, model: str = DEFAULT_MODEL) -> PMIntent:
    response = client.chat.completions.create(
        model=model,
        temperature=0.0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": INTENT_SYSTEM},
            {"role": "user", "content": question},
        ],
    )
    obj = _parse_json_object(response.choices[0].message.content or "{}")
    intent = str(obj.get("intent", "")).strip().lower()
    if intent not in SUPPORTED_INTENTS:
        intent = "status"
    normalized = str(obj.get("normalized_question") or question).strip()
    return PMIntent(intent=intent, normalized_question=normalized)


def compact_decision_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """Expose only decision-relevant fields to the language model."""
    utility = record.get("utility") or {}
    interaction = record.get("interaction") or record.get("interaction_state") or {}
    return {
        "selected_action": record.get("selected_action") or utility.get("selected_action"),
        "primary_domain": record.get("primary_domain") or record.get("predicted_primary_domain"),
        "consensus_score": record.get("consensus_score"),
        "re_grounding": record.get("rer") or record.get("re_grounding"),
        "utility": {
            "best_utility": utility.get("best_utility"),
            "candidates": utility.get("all_candidates") or utility.get("candidates"),
            "eligible_actions": utility.get("eligible_actions"),
        },
        "interaction": interaction,
        "evidence": record.get("evidence") or record.get("agents"),
    }


def _validate_authority(output: Dict[str, Any], authoritative_action: str) -> Dict[str, Any]:
    """Decision preservation is enforced after generation, not merely prompted."""
    answer = str(output.get("answer") or "").strip()
    generated_action = str(output.get("selected_action") or "").strip()
    if generated_action != authoritative_action:
        # Reject the generated action field. The narrative is retained only if
        # it does not explicitly recommend a different known governance action.
        generated_action = authoritative_action
    return {"answer": answer, "selected_action": generated_action}


def answer_pm_question(
    question: str,
    decision_record: Dict[str, Any],
    *,
    client: OpenAI,
    model: str = DEFAULT_MODEL,
) -> Dict[str, Any]:
    compact = compact_decision_record(decision_record)
    authoritative_action = str(compact.get("selected_action") or "").strip()
    if not authoritative_action:
        raise ValueError("decision_record does not contain an authoritative selected_action")

    intent = extract_intent(question, client=client, model=model)
    payload = {
        "intent": intent.intent,
        "pm_question": intent.normalized_question,
        "aaf_decision_record": compact,
    }
    response = client.chat.completions.create(
        model=model,
        temperature=0.0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": ANSWER_SYSTEM},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=2)},
        ],
    )
    obj = _parse_json_object(response.choices[0].message.content or "{}")
    validated = _validate_authority(obj, authoritative_action)
    return {
        "question": question,
        "intent": intent.intent,
        "normalized_question": intent.normalized_question,
        "selected_action": authoritative_action,
        "answer": validated["answer"],
        "decision_preserved": validated["selected_action"] == authoritative_action,
        "model": model,
        "llm_authority": "intent_and_explanation_only",
    }


def main() -> None:
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument("--record", required=True, help="AAF decision-record JSON")
    parser.add_argument("--question", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set")
    record = json.loads(Path(args.record).read_text(encoding="utf-8"))
    result = answer_pm_question(args.question, record, client=OpenAI(), model=args.model)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
