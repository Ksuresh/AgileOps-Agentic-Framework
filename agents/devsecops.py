from __future__ import annotations

from typing import Dict, Any, List
from .base import BaseAgent, AgentOutput


class DevSecOpsAgent(BaseAgent):
    agent_type = "DevSecOps"

    def infer(self, telemetry: Dict[str, Any]) -> AgentOutput:
        s = telemetry.get("sec", {}) or {}
        evidence: List[str] = []

        if s.get("_missing"):
            return AgentOutput(
                self.agent_type,
                "Security evidence is incomplete",
                0.45,
                ["Security telemetry marked as missing"],
            )

        cves = int(float(s.get("critical_cves", 0) or 0))
        policy = bool(s.get("policy_violation", False))
        iam = bool(s.get("iam_drift", False))
        compliance_gap = bool(s.get("compliance_gap", False))

        score = 0.0

        if cves >= 2:
            evidence.append(f"Multiple critical CVEs detected: {cves}")
            score += 0.40
        elif cves == 1:
            evidence.append("Critical CVE detected")
            score += 0.30

        if policy:
            evidence.append("Policy-as-code violation detected")
            score += 0.25

        if iam:
            evidence.append("IAM drift detected")
            score += 0.20

        if compliance_gap:
            evidence.append("Compliance evidence gap detected")
            score += 0.20

        if score >= 0.55:
            claim = "Security or compliance issue is the likely primary operational cause"
            confidence = min(0.92, 0.55 + score)
        elif score >= 0.30:
            claim = "Security signal indicates a possible policy or compliance risk"
            confidence = min(0.78, 0.50 + score)
        else:
            claim = "No material security or compliance anomaly detected"
            confidence = 0.25
            evidence = ["No critical CVE, policy violation, IAM drift, or compliance gap"]

        return AgentOutput(self.agent_type, claim, confidence, evidence)
