from __future__ import annotations

from typing import Dict, Any, List
from .base import BaseAgent, AgentOutput
from evidence.schema import bool_evidence, numeric_evidence, provenance_suffix


class DevSecOpsAgent(BaseAgent):
    agent_type = "DevSecOps"

    def infer(self, telemetry: Dict[str, Any]) -> AgentOutput:
        s = telemetry.get("sec", {}) or {}
        evidence: List[str] = []
        cve_ev = numeric_evidence(s, "critical_cves")
        policy_ev = bool_evidence(s, "policy_violation")
        iam_ev = bool_evidence(s, "iam_drift")
        compliance_ev = bool_evidence(s, "compliance_gap")
        available = [ev for ev in (cve_ev, policy_ev, iam_ev, compliance_ev) if ev.usable]
        if not available:
            return AgentOutput(self.agent_type, "Security evidence is incomplete", 0.45, ["No usable measured or proxy security evidence"])

        score = 0.0
        if cve_ev.usable:
            cves = int(float(cve_ev.value))
            if cves >= 2:
                evidence.append(f"Multiple critical CVEs detected: {cves}{provenance_suffix(cve_ev)}"); score += 0.40
            elif cves == 1:
                evidence.append(f"Critical CVE detected{provenance_suffix(cve_ev)}"); score += 0.30
        if policy_ev.usable and policy_ev.value:
            evidence.append(f"Policy-as-code violation detected{provenance_suffix(policy_ev)}"); score += 0.25
        if iam_ev.usable and iam_ev.value:
            evidence.append(f"IAM drift detected{provenance_suffix(iam_ev)}"); score += 0.20
        if compliance_ev.usable and compliance_ev.value:
            evidence.append(f"Compliance evidence gap detected{provenance_suffix(compliance_ev)}"); score += 0.20

        if score >= 0.55:
            claim = "Security or compliance issue is the likely primary operational cause"; confidence = min(0.92, 0.55 + score)
        elif score >= 0.30:
            claim = "Security signal indicates a possible policy or compliance risk"; confidence = min(0.78, 0.50 + score)
        else:
            claim = "No material security or compliance anomaly detected"; confidence = 0.25
            if not evidence: evidence = [f"No threshold breach in {len(available)} available security evidence field(s)"]
        return AgentOutput(self.agent_type, claim, confidence, evidence)
