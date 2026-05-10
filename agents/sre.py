from __future__ import annotations

from typing import Dict, Any, List
from .base import BaseAgent, AgentOutput


class SREAgent(BaseAgent):
    agent_type = "SRE"

    def infer(self, telemetry: Dict[str, Any]) -> AgentOutput:
        s = telemetry.get("sre", {}) or {}
        evidence: List[str] = []

        if s.get("_missing"):
            return AgentOutput(
                self.agent_type,
                "Reliability evidence is incomplete",
                0.45,
                ["SRE telemetry marked as missing"],
            )

        p95 = float(s.get("p95_latency_ms", 0.0) or 0.0)
        err = float(s.get("error_rate_pct", 0.0) or 0.0)
        sat = float(s.get("saturation_pct", 0.0) or 0.0)
        availability = float(s.get("availability_pct", 99.9) or 99.9)

        score = 0.0

        if p95 >= 800:
            evidence.append(f"Severe P95 latency elevation: {p95:.0f} ms")
            score += 0.35
        elif p95 >= 450:
            evidence.append(f"P95 latency elevated: {p95:.0f} ms")
            score += 0.25

        if err >= 12:
            evidence.append(f"Severe error rate elevation: {err:.1f}%")
            score += 0.35
        elif err >= 8:
            evidence.append(f"Error rate elevated: {err:.1f}%")
            score += 0.25

        if sat >= 90:
            evidence.append(f"Severe saturation level: {sat:.0f}%")
            score += 0.25
        elif sat >= 85:
            evidence.append(f"Saturation elevated: {sat:.0f}%")
            score += 0.15

        if availability < 99.0:
            evidence.append(f"Availability dropped to {availability:.2f}%")
            score += 0.25

        if score >= 0.60:
            claim = "Reliability degradation is the likely primary operational cause"
            confidence = min(0.92, 0.55 + score)
        elif score >= 0.30:
            claim = "Reliability signal indicates a possible service health issue"
            confidence = min(0.78, 0.50 + score)
        else:
            claim = "No material reliability anomaly detected"
            confidence = 0.25
            evidence = ["Latency, error rate, saturation, and availability are within expected range"]

        return AgentOutput(self.agent_type, claim, confidence, evidence)
