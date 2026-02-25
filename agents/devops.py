from __future__ import annotations
from typing import Dict, Any, List
from .base import BaseAgent, AgentOutput

class DevOpsAgent(BaseAgent):
    agent_type = "DevOps"

    def infer(self, telemetry: Dict[str, Any]) -> AgentOutput:
        d = telemetry.get("deploy", {})
        evidence: List[str] = []
        conf = 0.55

        if d.get("_missing"):
            return AgentOutput(self.agent_type, "Deployment signal unavailable", 0.35, ["Deploy telemetry missing"])

        if d.get("pipeline_failed"):
            evidence.append("CI/CD pipeline failure detected")
            conf += 0.15
        if d.get("config_drift"):
            evidence.append("Configuration drift indicator present")
            conf += 0.10
        restarts = int(d.get("restart_loops", 0))
        if restarts >= 12:
            evidence.append(f"Restart loops observed: {restarts} in short window")
            conf += 0.15

        if evidence:
            claim = "Deployment instability likely due to pipeline/config issues"
        else:
            claim = "No deployment anomaly detected"
            conf = 0.70
            evidence = ["No pipeline failure", "No config drift", "No abnormal restarts"]

        return AgentOutput(self.agent_type, claim, min(conf, 0.95), evidence)
