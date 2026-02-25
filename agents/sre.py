from __future__ import annotations
from typing import Dict, Any, List
from .base import BaseAgent, AgentOutput

class SREAgent(BaseAgent):
    agent_type = "SRE"

    def infer(self, telemetry: Dict[str, Any]) -> AgentOutput:
        s = telemetry.get("sre", {})
        evidence: List[str] = []
        conf = 0.55
        if s.get("_missing"):
            return AgentOutput(self.agent_type, "Reliability signal unavailable", 0.35, ["SRE telemetry missing"])

        p95 = float(s.get("p95_latency_ms", 0.0))
        err = float(s.get("error_rate_pct", 0.0))
        sat = float(s.get("saturation_pct", 0.0))

        if p95 >= 450:
            evidence.append(f"P95 latency elevated: {p95:.0f} ms"); conf += 0.15
        if err >= 8.0:
            evidence.append(f"Error rate elevated: {err:.1f}%"); conf += 0.15
        if sat >= 85:
            evidence.append(f"Saturation elevated: {sat:.0f}%"); conf += 0.10

        if evidence:
            claim = "Service reliability degradation detected (latency/error/saturation)"
        else:
            claim = "No reliability anomaly detected"
            conf = 0.70
            evidence = ["Latency within expected range", "Error rate normal", "Saturation normal"]

        return AgentOutput(self.agent_type, claim, min(conf, 0.95), evidence)
