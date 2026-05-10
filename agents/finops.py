from __future__ import annotations

from typing import Dict, Any, List
from .base import BaseAgent, AgentOutput


class FinOpsAgent(BaseAgent):
    agent_type = "FinOps"

    def infer(self, telemetry: Dict[str, Any]) -> AgentOutput:
        f = telemetry.get("finops", {}) or {}
        evidence: List[str] = []

        if f.get("_missing"):
            return AgentOutput(
                self.agent_type,
                "Cost evidence is incomplete",
                0.45,
                ["FinOps telemetry marked as missing"],
            )

        spike = float(f.get("cost_spike_pct", 0.0) or 0.0)
        hpa = int(float(f.get("hpa_scale_to", 0) or 0))
        cpu_inc = float(f.get("cpu_request_increase_pct", 0.0) or 0.0)
        mem_inc = float(f.get("memory_request_increase_pct", 0.0) or 0.0)

        score = 0.0

        if spike >= 35:
            evidence.append(f"Severe cost spike detected: {spike:.0f}%")
            score += 0.40
        elif spike >= 22:
            evidence.append(f"Cost spike detected: {spike:.0f}%")
            score += 0.30

        if hpa >= 14:
            evidence.append(f"Large HPA scale-out observed: {hpa} pods")
            score += 0.25
        elif hpa >= 11:
            evidence.append(f"HPA scale-out observed: {hpa} pods")
            score += 0.15

        if cpu_inc >= 50:
            evidence.append(f"CPU request increase detected: {cpu_inc:.0f}%")
            score += 0.20

        if mem_inc >= 40:
            evidence.append(f"Memory request increase detected: {mem_inc:.0f}%")
            score += 0.20

        if score >= 0.55:
            claim = "Cost or resource efficiency issue is the likely primary operational cause"
            confidence = min(0.92, 0.55 + score)
        elif score >= 0.30:
            claim = "Cost signal indicates a possible scaling or provisioning issue"
            confidence = min(0.78, 0.50 + score)
        else:
            claim = "No material cost anomaly detected"
            confidence = 0.25
            evidence = ["No significant cost spike, scale-out, or resource request increase"]

        return AgentOutput(self.agent_type, claim, confidence, evidence)
