from __future__ import annotations

from typing import Dict, Any, List
from .base import BaseAgent, AgentOutput


class DevOpsAgent(BaseAgent):
    agent_type = "DevOps"

    def infer(self, telemetry: Dict[str, Any]) -> AgentOutput:
        d = telemetry.get("deploy", {}) or {}
        evidence: List[str] = []

        if d.get("_missing"):
            return AgentOutput(
                self.agent_type,
                "Deployment evidence is incomplete",
                0.45,
                ["Deployment telemetry marked as missing"],
            )

        pipeline_failed = bool(d.get("pipeline_failed", False))
        config_drift = bool(d.get("config_drift", False))
        rollback_marker = bool(d.get("rollback_marker", False))
        artifact_mismatch = bool(d.get("artifact_mismatch", False))
        restart_loops = int(float(d.get("restart_loops", 0) or 0))

        score = 0.0

        if pipeline_failed:
            evidence.append("CI/CD pipeline failure detected")
            score += 0.30
        if config_drift:
            evidence.append("Configuration drift detected")
            score += 0.25
        if rollback_marker:
            evidence.append("Rollback marker present in release telemetry")
            score += 0.25
        if artifact_mismatch:
            evidence.append("Deployment artifact mismatch detected")
            score += 0.25
        if restart_loops >= 12:
            evidence.append(f"Container restart loops observed: {restart_loops}")
            score += 0.25
        elif restart_loops >= 6:
            evidence.append(f"Moderate restart loops observed: {restart_loops}")
            score += 0.12

        if score >= 0.60:
            claim = "Deployment failure is the likely primary operational cause"
            confidence = min(0.92, 0.55 + score)
        elif score >= 0.30:
            claim = "Deployment signal indicates a possible release or configuration issue"
            confidence = min(0.78, 0.50 + score)
        else:
            claim = "No material deployment anomaly detected"
            confidence = 0.25
            evidence = ["No pipeline failure, rollback marker, artifact mismatch, or abnormal restart loop"]

        return AgentOutput(self.agent_type, claim, confidence, evidence)
