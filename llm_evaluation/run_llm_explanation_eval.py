from __future__ import annotations

import argparse
import csv
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List

from openai import OpenAI

from metrics.explainability import compute_xi


DEFAULT_INPUT = Path("results_final_phase2a_pm_prompts/pm_prompt_outputs.jsonl")
DEFAULT_OUT_DIR = Path("results_phase2b_llm_explanations")
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# Update these if you use a different model/pricing.
# Values are USD per 1M tokens. These are placeholders for reproducible local cost estimation.
MODEL_PRICING_PER_1M = {
    "gpt-4o-mini": {
        "input": 0.15,
        "output": 0.60,
    }
}


SYSTEM_PROMPT = """You are a Project Manager-facing AgileOps governance assistant.

Your task is to explain the governance decision using only the structured evidence provided.

Rules:
- Do not invent facts.
- Do not introduce telemetry, incidents, risks, or causes that are not present in the input.
- Preserve the selected action exactly as provided.
- Preserve the predicted primary domain exactly as provided.
- If evidence is missing or incomplete, state that it is incomplete.
- Use clear business-facing language for Project Managers.
- Keep the explanation concise.
"""


USER_PROMPT_TEMPLATE = """Generate a Project Manager-facing governance explanation from the structured AAF output below.

Required output format:

1. What happened
2. Why it happened
3. Cross-domain impact
4. Recommended governance action
5. Confidence and uncertainty
6. Evidence used
7. PM decision implication

Structured AAF output:
{payload_json}
"""


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _compact_case(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "scenario_id": row.get("scenario_id"),
        "pm_prompt": row.get("prompt"),
        "expected_domains": (row.get("ground_truth") or {}).get("expected_domains"),
        "expected_action": (row.get("ground_truth") or {}).get("expected_action"),
        "predicted_primary_domain": row.get("predicted_primary_domain"),
        "selected_action": (row.get("utility") or {}).get("selected_action"),
        "consensus_score": row.get("consensus_score"),
        "rar": row.get("rar"),
        "utility": row.get("utility"),
        "agents": row.get("agents"),
        "explainability": row.get("explainability"),
    }


def _call_llm(client: OpenAI, model: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    payload_json = json.dumps(payload, indent=2, ensure_ascii=False)
    user_prompt = USER_PROMPT_TEMPLATE.format(payload_json=payload_json)

    response = client.chat.completions.create(
        model=model,
        temperature=0.0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )

    text = response.choices[0].message.content or ""

    usage = response.usage
    prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
    completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
    total_tokens = int(getattr(usage, "total_tokens", 0) or 0)

    return {
        "llm_explanation": text,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


def _normalize_action(action: str | None) -> str:
    if not action:
        return ""

    a = str(action).lower().strip()

    if "rollback" in a:
        return "rollback"
    if "patch" in a or "block" in a or "security" in a:
        return "patch_block"
    if "scale" in a or "scaling" in a:
        return "scale"
    if "mitigate" in a or "monitor" in a:
        return "mitigate_monitor"
    if "review" in a:
        return "review"
    if "observe" in a or "no action" in a:
        return "observe"
    if "defer" in a:
        return "defer"

    return a


def _action_consistent(text: str, selected_action: str | None) -> bool:
    if not selected_action:
        return False
    return _normalize_action(selected_action) in _normalize_action(text)


def _domain_consistent(text: str, domain: str | None) -> bool:
    if not domain:
        return False
    return str(domain).lower() in text.lower()


def _evidence_terms(row: Dict[str, Any]) -> List[str]:
    terms: List[str] = []

    for agent in row.get("agents", []) or []:
        agent_type = agent.get("agent_type")
        if agent_type:
            terms.append(str(agent_type).lower())

        for ev in agent.get("evidence") or []:
            if isinstance(ev, str) and ev.strip():
                terms.append(ev.strip().lower())
                for token in re.findall(r"[A-Za-z][A-Za-z\-]+", ev.lower()):
                    if len(token) >= 6:
                        terms.append(token)

    return sorted(set(terms))


def _evidence_coverage(text: str, row: Dict[str, Any]) -> float:
    terms = _evidence_terms(row)
    if not terms:
        return 0.0

    text_l = text.lower()
    hits = sum(1 for term in terms if term in text_l)
    target = min(10, max(4, len(terms) // 3))

    return min(1.0, hits / target)


def _unsupported_claim_risk(text: str, row: Dict[str, Any]) -> float:
    """
    Lightweight reproducible proxy, not a full factuality judge.

    It flags whether the LLM introduces terms outside the expected governance vocabulary
    and provided evidence. Lower is better.
    """
    allowed_terms = set(
        [
            "devops",
            "sre",
            "finops",
            "devsecops",
            "deployment",
            "release",
            "latency",
            "error",
            "saturation",
            "availability",
            "cost",
            "scale",
            "scaling",
            "security",
            "policy",
            "compliance",
            "cve",
            "iam",
            "rollback",
            "patch",
            "block",
            "monitor",
            "risk",
            "evidence",
            "consensus",
            "utility",
            "project",
            "manager",
            "governance",
            "action",
            "confidence",
            "uncertainty",
            "prompt",
            "telemetry",
            "agent",
        ]
    )

    for term in _evidence_terms(row):
        for token in re.findall(r"[a-z][a-z\-]+", term.lower()):
            if len(token) >= 4:
                allowed_terms.add(token)

    words = [
        w.lower()
        for w in re.findall(r"[A-Za-z][A-Za-z\-]+", text)
        if len(w) >= 7
    ]

    if not words:
        return 0.0

    unsupported = [w for w in words if w not in allowed_terms]

    # Cap to avoid over-penalizing normal prose.
    return min(1.0, len(unsupported) / max(20, len(words)))


def _estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    pricing = MODEL_PRICING_PER_1M.get(model, MODEL_PRICING_PER_1M["gpt-4o-mini"])
    return (
        (prompt_tokens / 1_000_000.0) * pricing["input"]
        + (completion_tokens / 1_000_000.0) * pricing["output"]
    )


def _score_output(row: Dict[str, Any], llm_text: str, model: str, token_data: Dict[str, int]) -> Dict[str, Any]:
    utility = row.get("utility") or {}

    selected_action = utility.get("selected_action")
    predicted_domain = row.get("predicted_primary_domain")

    xi_payload = {
        "agents": row.get("agents", []),
    }
    xi = compute_xi(llm_text, xi_payload)

    prompt_tokens = int(token_data.get("prompt_tokens", 0))
    completion_tokens = int(token_data.get("completion_tokens", 0))

    return {
        "domain_consistent": _domain_consistent(llm_text, predicted_domain),
        "action_consistent": _action_consistent(llm_text, selected_action),
        "evidence_coverage": _evidence_coverage(llm_text, row),
        "unsupported_claim_risk": _unsupported_claim_risk(llm_text, row),
        "xi": xi,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": int(token_data.get("total_tokens", prompt_tokens + completion_tokens)),
        "estimated_cost_usd": _estimate_cost(model, prompt_tokens, completion_tokens),
    }


def _mean(values: List[float | bool]) -> float:
    if not values:
        return 0.0
    return float(sum(float(v) for v in values) / len(values))


def _summarize(outputs: List[Dict[str, Any]], model: str) -> Dict[str, Any]:
    scores = [o.get("scores", {}) for o in outputs]

    return {
        "n": len(outputs),
        "model": model,
        "temperature": 0.0,
        "domain_consistency_rate": _mean([s.get("domain_consistent", False) for s in scores]),
        "action_consistency_rate": _mean([s.get("action_consistent", False) for s in scores]),
        "evidence_coverage_mean": _mean([s.get("evidence_coverage", 0.0) for s in scores]),
        "unsupported_claim_risk_mean": _mean([s.get("unsupported_claim_risk", 0.0) for s in scores]),
        "xi_mean": _mean([(s.get("xi") or {}).get("xi", 0.0) for s in scores]),
        "readability_mean": _mean([(s.get("xi") or {}).get("readability", 0.0) for s in scores]),
        "evidence_clarity_mean": _mean([(s.get("xi") or {}).get("evidence_clarity", 0.0) for s in scores]),
        "traceability_mean": _mean([(s.get("xi") or {}).get("traceability", 0.0) for s in scores]),
        "prompt_tokens_total": sum(int(s.get("prompt_tokens", 0)) for s in scores),
        "completion_tokens_total": sum(int(s.get("completion_tokens", 0)) for s in scores),
        "tokens_total": sum(int(s.get("total_tokens", 0)) for s in scores),
        "estimated_cost_usd_total": sum(float(s.get("estimated_cost_usd", 0.0)) for s in scores),
        "pricing_note": "Cost estimate uses local pricing table in run_llm_explanation_eval.py; update table if model pricing changes.",
    }


def _write_cost_csv(path: Path, outputs: List[Dict[str, Any]]) -> None:
    fieldnames = [
        "scenario_id",
        "model",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "estimated_cost_usd",
        "domain_consistent",
        "action_consistent",
        "evidence_coverage",
        "unsupported_claim_risk",
        "xi",
    ]

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for o in outputs:
            s = o.get("scores", {})
            xi = s.get("xi") or {}

            writer.writerow(
                {
                    "scenario_id": o.get("scenario_id"),
                    "model": o.get("model"),
                    "prompt_tokens": s.get("prompt_tokens"),
                    "completion_tokens": s.get("completion_tokens"),
                    "total_tokens": s.get("total_tokens"),
                    "estimated_cost_usd": s.get("estimated_cost_usd"),
                    "domain_consistent": s.get("domain_consistent"),
                    "action_consistent": s.get("action_consistent"),
                    "evidence_coverage": s.get("evidence_coverage"),
                    "unsupported_claim_risk": s.get("unsupported_claim_risk"),
                    "xi": xi.get("xi"),
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--out", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.out)
    model = args.model

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set. Set it as an environment variable before running.")

    out_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_jsonl(input_path)
    selected_rows = rows[: args.limit]

    client = OpenAI()

    outputs: List[Dict[str, Any]] = []

    for idx, row in enumerate(selected_rows, start=1):
        compact = _compact_case(row)
        llm_result = _call_llm(client, model, compact)

        token_data = {
            "prompt_tokens": llm_result["prompt_tokens"],
            "completion_tokens": llm_result["completion_tokens"],
            "total_tokens": llm_result["total_tokens"],
        }

        scores = _score_output(row, llm_result["llm_explanation"], model, token_data)

        outputs.append(
            {
                "index": idx,
                "scenario_id": row.get("scenario_id"),
                "model": model,
                "temperature": 0.0,
                "pm_prompt": row.get("prompt"),
                "predicted_primary_domain": row.get("predicted_primary_domain"),
                "selected_action": (row.get("utility") or {}).get("selected_action"),
                "llm_explanation": llm_result["llm_explanation"],
                "scores": scores,
                "structured_input": compact,
            }
        )

        print(f"[{idx}/{len(selected_rows)}] {row.get('scenario_id')} done")

    summary = _summarize(outputs, model)

    _write_jsonl(out_dir / "llm_explanation_outputs.jsonl", outputs)
    _write_cost_csv(out_dir / "llm_cost_summary.csv", outputs)

    with open(out_dir / "llm_explanation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with open(out_dir / "llm_prompt_template.md", "w", encoding="utf-8") as f:
        f.write(SYSTEM_PROMPT)
        f.write("\n\n")
        f.write(USER_PROMPT_TEMPLATE)

    print(json.dumps(summary, indent=2))
    print(f"Wrote LLM explanation evaluation to: {out_dir}")


if __name__ == "__main__":
    main()
