from __future__ import annotations
from typing import Dict, Any, List
from .base import BaseAgent, AgentOutput

class FinOpsAgent(BaseAgent):
    agent_type = "FinOps"

    def infer(self, telemetry: Dict[str, Any]) -> AgentOutput:
        f = telemetry.get("finops", {})
        evidence: List[str] = []
        conf = 0.55
        if f.get("_missing"):
            return AgentOutput(self.agent_type, "Cost signal unavailable", 0.35, ["FinOps telemetry missing"])

        spike = float(f.get("cost_spike_pct", 0.0))
        hpa = int(f.get("hpa_scale_to", 0))

        if spike >= 22:
            evidence.append(f"Cost spike detected: {spike:.0f}%"); conf += 0.20
        if hpa >= 11:
            evidence.append(f"HPA scale-out observed: scaled to {hpa} pods"); conf += 0.10

        if evidence:
            claim = "Cost anomaly likely driven by scaling/provisioning behavior"
        else:
            claim = "No cost anomaly detected"
            conf = 0.70
            evidence = ["No cost spike", "No unusual scaling"]

        return AgentOutput(self.agent_type, claim, min(conf, 0.95), evidence)
