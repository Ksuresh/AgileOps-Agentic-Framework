from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class AgentOutput:
    agent_type: str
    claim: str
    confidence: float
    evidence: List[str]

class BaseAgent:
    agent_type: str = "Base"
    def infer(self, telemetry: Dict[str, Any]) -> AgentOutput:
        raise NotImplementedError
