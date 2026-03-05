"""Deterministic explanation generator (evidence-grounded).

This module is a *drop-in* replacement for an actual LLM call during
development/review.

Why include it?
- Reviewers may ask to reproduce tables quickly.
- The repo should produce consistent outputs without external APIs.

If you later run a real open-weight model (e.g., Mistral 7B via llama.cpp),
swap this function with your inference wrapper while keeping the same prompt
schema from Appendix A.
"""

from __future__ import annotations

from typing import Dict, Any, List


SYSTEM_PROMPT = (
    "You are an operational governance assistant for Agile--DevOps environments. "
    "Use ONLY the structured evidence provided. Do NOT speculate or introduce new facts. "
    "Generate a structured explanation with five sections: What happened; Why; Impact; "
    "Recommended action; Confidence. Use plain language suitable for a project manager."
)


def generate_explanation(payload: Dict[str, Any]) -> str:
    """Produce a deterministic, evidence-only explanation."""
    agents: List[Dict[str, Any]] = payload.get("agents", [])
    cons = payload.get("consensus_score")
    action = payload.get("recommended_action")

    # Evidence is intentionally terse. The goal is to be stable, grounded, and readable.
    devops = _agent(agents, "DevOps")
    sre = _agent(agents, "SRE")
    fin = _agent(agents, "FinOps")
    sec = _agent(agents, "DevSecOps")

    happened = _first_non_empty([
        devops.get("claim"),
        sre.get("claim"),
        fin.get("claim"),
        sec.get("claim"),
    ])

    why = _join_non_empty([
        _reason_line("DevOps", devops),
        _reason_line("SRE", sre),
        _reason_line("FinOps", fin),
        _reason_line("DevSecOps", sec),
    ])

    impact = _join_non_empty([
        _impact_line("Performance", sre),
        _impact_line("Cost", fin),
        _impact_line("Security/Compliance", sec),
        _impact_line("Deployment", devops),
    ])

    conf_label = "High" if (cons is not None and cons >= 0.75) else "Medium" if (cons is not None and cons >= 0.55) else "Low"

    return (
        "1. What Happened:\n"
        f"{happened or 'An operational anomaly was detected based on the provided evidence.'}\n\n"
        "2. Why It Happened:\n"
        f"{why or 'Agents reported limited or non-overlapping evidence.'}\n\n"
        "3. Impact Across Domains:\n"
        f"{impact or 'Impact signals were limited in the available evidence.'}\n\n"
        "4. Recommended Action:\n"
        f"{action or 'Defer and request additional evidence.'}\n\n"
        "5. Confidence Level:\n"
        f"{conf_label} confidence (consensus score: {cons:.2f})." if cons is not None else f"{conf_label} confidence."
    )


def _agent(agents: List[Dict[str, Any]], typ: str) -> Dict[str, Any]:
    for a in agents:
        if a.get("agent_type") == typ:
            return a
    return {}


def _first_non_empty(xs: List[str | None]) -> str:
    for x in xs:
        if x and x.strip():
            return x.strip()
    return ""


def _join_non_empty(lines: List[str]) -> str:
    lines = [l.strip() for l in lines if l and l.strip()]
    return " ".join(lines)


def _reason_line(label: str, agent: Dict[str, Any]) -> str:
    claim = (agent.get("claim") or "").strip()
    ev = agent.get("evidence") or []
    ev_snip = f" Evidence: {ev[0]}." if ev else ""
    if not claim:
        return ""
    return f"{label}: {claim}.{ev_snip}"


def _impact_line(label: str, agent: Dict[str, Any]) -> str:
    ev = agent.get("evidence") or []
    if not ev:
        return ""
    return f"{label}: {ev[0]}."
