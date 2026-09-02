from __future__ import annotations

from typing import Dict, Any, List
from .base import BaseAgent, AgentOutput
from evidence.schema import numeric_evidence, provenance_suffix


class FinOpsAgent(BaseAgent):
    agent_type = "FinOps"

    def infer(self, telemetry: Dict[str, Any]) -> AgentOutput:
        f = telemetry.get("finops", {}) or {}
        evidence: List[str] = []
        spike_ev = numeric_evidence(f, "cost_spike_pct")
        hpa_ev = numeric_evidence(f, "hpa_scale_to")
        cpu_ev = numeric_evidence(f, "cpu_request_increase_pct")
        mem_ev = numeric_evidence(f, "memory_request_increase_pct")
        available = [ev for ev in (spike_ev, hpa_ev, cpu_ev, mem_ev) if ev.usable]
        if not available:
            return AgentOutput(self.agent_type, "Cost/resource evidence is incomplete", 0.45, ["No usable measured or proxy FinOps evidence"])

        score = 0.0
        if spike_ev.usable:
            spike = float(spike_ev.value)
            if spike >= 35:
                evidence.append(f"Severe cost/resource proxy increase: {spike:.0f}%{provenance_suffix(spike_ev)}"); score += 0.40
            elif spike >= 22:
                evidence.append(f"Cost/resource proxy increase: {spike:.0f}%{provenance_suffix(spike_ev)}"); score += 0.30
        if hpa_ev.usable:
            hpa = int(float(hpa_ev.value))
            if hpa >= 14:
                evidence.append(f"Large HPA scale-out observed: {hpa} pods{provenance_suffix(hpa_ev)}"); score += 0.25
            elif hpa >= 11:
                evidence.append(f"HPA scale-out observed: {hpa} pods{provenance_suffix(hpa_ev)}"); score += 0.15
        if cpu_ev.usable and float(cpu_ev.value) >= 50:
            evidence.append(f"CPU request increase detected: {float(cpu_ev.value):.0f}%{provenance_suffix(cpu_ev)}"); score += 0.20
        if mem_ev.usable and float(mem_ev.value) >= 40:
            evidence.append(f"Memory request increase detected: {float(mem_ev.value):.0f}%{provenance_suffix(mem_ev)}"); score += 0.20

        if score >= 0.55:
            claim = "Cost or resource efficiency issue is the likely primary operational cause"; confidence = min(0.92, 0.55 + score)
        elif score >= 0.30:
            claim = "Cost signal indicates a possible scaling or provisioning issue"; confidence = min(0.78, 0.50 + score)
        else:
            claim = "No material cost/resource anomaly detected"; confidence = 0.25
            if not evidence: evidence = [f"No threshold breach in {len(available)} available FinOps evidence field(s)"]
        return AgentOutput(self.agent_type, claim, confidence, evidence)
