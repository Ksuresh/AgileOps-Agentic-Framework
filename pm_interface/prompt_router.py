from __future__ import annotations

from typing import Dict, List


KEYWORD_MAP = {
    "deployment_failure": [
        "failed deployment",
        "deployment failed",
        "release failed",
        "rollback",
        "pipeline failed",
        "build failed",
        "config drift",
        "restart loop",
    ],
    "latency_spike": [
        "latency spike",
        "slow response",
        "timeout",
        "timeouts",
        "error rate",
        "service degraded",
        "slo breach",
        "availability issue",
    ],
    "cost_spike": [
        "cost spike",
        "cost increased",
        "billing issue",
        "cloud spend",
        "budget overrun",
        "too expensive",
        "scale cost",
    ],
    "security_violation": [
        "security issue",
        "policy violation",
        "compliance issue",
        "cve",
        "vulnerability",
        "iam drift",
        "security flagged",
    ],
}


def _contains_any(text: str, phrases: List[str]) -> bool:
    t = text.lower()
    return any(p in t for p in phrases)


def route_prompt(prompt: str) -> Dict:
    text = (prompt or "").strip().lower()

    symptoms: List[str] = []
    for symptom, phrases in KEYWORD_MAP.items():
        if _contains_any(text, phrases):
            symptoms.append(symptom)

    domains: List[str] = []
    if "deployment_failure" in symptoms:
        domains.append("DevOps")
    if "latency_spike" in symptoms:
        domains.append("SRE")
    if "cost_spike" in symptoms:
        domains.append("FinOps")
    if "security_violation" in symptoms:
        domains.append("DevSecOps")

    if not symptoms:
        symptoms = ["general_investigation"]

    if not domains:
        domains = ["DevOps", "SRE", "FinOps", "DevSecOps"]

    priority = "medium"
    if any(x in text for x in ["urgent", "critical", "sev1", "sev-1", "high priority"]):
        priority = "high"
    elif any(x in text for x in ["low priority", "minor", "sev3", "sev-3"]):
        priority = "low"

    return {
        "intent": "operational_governance_decision",
        "prompt": prompt,
        "symptoms": symptoms,
        "suspected_domains": domains,
        "priority": priority,
    }
