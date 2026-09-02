from __future__ import annotations

from typing import Dict, Any, List
from .base import BaseAgent, AgentOutput
from evidence.schema import numeric_evidence, provenance_suffix


class SREAgent(BaseAgent):
    agent_type = "SRE"

    def infer(self, telemetry: Dict[str, Any]) -> AgentOutput:
        s = telemetry.get("sre", {}) or {}
        evidence: List[str] = []

        p95_ev = numeric_evidence(s, "p95_latency_ms")
        err_ev = numeric_evidence(s, "error_rate_pct")
        sat_ev = numeric_evidence(s, "saturation_pct")
        availability_ev = numeric_evidence(s, "availability_pct")
        available = [ev for ev in (p95_ev, err_ev, sat_ev, availability_ev) if ev.usable]

        if not available:
            return AgentOutput(self.agent_type, "Reliability evidence is incomplete", 0.45, ["No usable measured or proxy SRE evidence"])

        score = 0.0
        if p95_ev.usable:
            p95 = float(p95_ev.value)
            if p95 >= 800:
                evidence.append(f"Severe P95 latency elevation: {p95:.0f} ms{provenance_suffix(p95_ev)}"); score += 0.35
            elif p95 >= 450:
                evidence.append(f"P95 latency elevated: {p95:.0f} ms{provenance_suffix(p95_ev)}"); score += 0.25
        if err_ev.usable:
            err = float(err_ev.value)
            if err >= 12:
                evidence.append(f"Severe error rate elevation: {err:.1f}%{provenance_suffix(err_ev)}"); score += 0.35
            elif err >= 8:
                evidence.append(f"Error rate elevated: {err:.1f}%{provenance_suffix(err_ev)}"); score += 0.25
        if sat_ev.usable:
            sat = float(sat_ev.value)
            if sat >= 90:
                evidence.append(f"Severe saturation level: {sat:.0f}%{provenance_suffix(sat_ev)}"); score += 0.25
            elif sat >= 85:
                evidence.append(f"Saturation elevated: {sat:.0f}%{provenance_suffix(sat_ev)}"); score += 0.15
        if availability_ev.usable:
            availability = float(availability_ev.value)
            if availability < 95.0:
                evidence.append(f"Severe availability indicator drop: {availability:.2f}%{provenance_suffix(availability_ev)}"); score += 0.35
            elif availability < 99.0:
                evidence.append(f"Availability indicator dropped to {availability:.2f}%{provenance_suffix(availability_ev)}"); score += 0.25

        if score >= 0.60:
            claim = "Reliability degradation is the likely primary operational cause"; confidence = min(0.92, 0.55 + score)
        elif score >= 0.30:
            claim = "Reliability signal indicates a possible service health issue"; confidence = min(0.78, 0.50 + score)
        else:
            claim = "No material reliability anomaly detected"; confidence = 0.25
            if not evidence: evidence = [f"No threshold breach in {len(available)} available SRE evidence field(s)"]
        return AgentOutput(self.agent_type, claim, confidence, evidence)
