from __future__ import annotations

from typing import Dict, Any, List
from .base import BaseAgent, AgentOutput
from evidence.schema import bool_evidence, numeric_evidence, provenance_suffix


class DevOpsAgent(BaseAgent):
    agent_type = "DevOps"

    def infer(self, telemetry: Dict[str, Any]) -> AgentOutput:
        d = telemetry.get("deploy", {}) or {}
        evidence: List[str] = []
        pipeline_ev = bool_evidence(d, "pipeline_failed")
        config_ev = bool_evidence(d, "config_drift")
        rollback_ev = bool_evidence(d, "rollback_marker")
        artifact_ev = bool_evidence(d, "artifact_mismatch")
        restart_ev = numeric_evidence(d, "restart_loops")
        available = [ev for ev in (pipeline_ev, config_ev, rollback_ev, artifact_ev, restart_ev) if ev.usable]
        if not available:
            return AgentOutput(self.agent_type, "Deployment evidence is incomplete", 0.45, ["No usable measured or proxy deployment evidence"])

        score = 0.0
        if pipeline_ev.usable and pipeline_ev.value:
            evidence.append(f"CI/CD pipeline failure detected{provenance_suffix(pipeline_ev)}"); score += 0.30
        if config_ev.usable and config_ev.value:
            evidence.append(f"Configuration drift detected{provenance_suffix(config_ev)}"); score += 0.25
        if rollback_ev.usable and rollback_ev.value:
            evidence.append(f"Rollback marker present in release telemetry{provenance_suffix(rollback_ev)}"); score += 0.25
        if artifact_ev.usable and artifact_ev.value:
            evidence.append(f"Deployment artifact mismatch detected{provenance_suffix(artifact_ev)}"); score += 0.25
        if restart_ev.usable:
            restart_loops = int(float(restart_ev.value))
            if restart_loops >= 12:
                evidence.append(f"Container restart loops observed: {restart_loops}{provenance_suffix(restart_ev)}"); score += 0.25
            elif restart_loops >= 6:
                evidence.append(f"Moderate restart loops observed: {restart_loops}{provenance_suffix(restart_ev)}"); score += 0.12

        if score >= 0.60:
            claim = "Deployment failure is the likely primary operational cause"; confidence = min(0.92, 0.55 + score)
        elif score >= 0.30:
            claim = "Deployment signal indicates a possible release or configuration issue"; confidence = min(0.78, 0.50 + score)
        else:
            claim = "No material deployment anomaly detected"; confidence = 0.25
            if not evidence: evidence = [f"No threshold breach in {len(available)} available deployment evidence field(s)"]
        return AgentOutput(self.agent_type, claim, confidence, evidence)
