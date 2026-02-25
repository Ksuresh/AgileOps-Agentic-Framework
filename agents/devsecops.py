from __future__ import annotations
from typing import Dict, Any, List
from .base import BaseAgent, AgentOutput

class DevSecOpsAgent(BaseAgent):
    agent_type = "DevSecOps"

    def infer(self, telemetry: Dict[str, Any]) -> AgentOutput:
        s = telemetry.get("sec", {})
        evidence: List[str] = []
        conf = 0.55
        if s.get("_missing"):
            return AgentOutput(self.agent_type, "Security signal unavailable", 0.35, ["Security telemetry missing"])

        cves = int(s.get("critical_cves", 0))
        pol = bool(s.get("policy_violation", False))
        iam = bool(s.get("iam_drift", False))

        if cves > 0:
            evidence.append(f"Critical CVEs detected: {cves}"); conf += 0.20
        if pol:
            evidence.append("Policy-as-code violation detected"); conf += 0.15
        if iam:
            evidence.append("IAM drift indicator detected"); conf += 0.10

        if evidence:
            claim = "Security/compliance risk detected requiring triage"
        else:
            claim = "No security or policy anomaly detected"
            conf = 0.80
            evidence = ["No critical CVEs", "No policy violations", "No IAM drift"]

        return AgentOutput(self.agent_type, claim, min(conf, 0.95), evidence)
