# Thresholds actually used in the code (checked 2026-10-22)

| Symbol | Where used | Value in code | Meaning |
| --- | --- | --- | --- |
| τ (re-grounding, RER) | `orchestrator/rar.py` default; per-scenario `thresholds.tau_consensus` in `scenario_generator/generate.py` and `reviewer_aligned.py` (controlled studies) | **0.65** | consensus below τ starts re-evaluation of available evidence (RER) |
| τ_AEI (selective investigation trigger) | `agentic_experiments/hybrid.py` `TAU` (32 + 16 selective-AEI studies) | **0.75** | decision readiness below τ_AEI (with a material signal) is one trigger for LLM evidence investigation |
| δ_min | `orchestrator/rar.py` default; per-scenario `thresholds.delta_min` (controlled studies) | **0.05** | minimum consensus improvement for RER to be accepted without escalation; it is **not** a materiality threshold |
| legacy config | `config/config.yaml` (artifact of the earlier submission Access-2025-54890) | τ 0.75, δ_min 0.15 | not used where scenarios carry their own thresholds (`pipeline.py` reads `scenario["thresholds"]` first) |
| materiality | `orchestrator/decision_gate.py`, `orchestrator/severity.py` | a domain is material when its severity score > 0, i.e. at least one field crosses its fixed absolute threshold (e.g. saturation ≥ 85%, p95 ≥ 450 ms, error rate ≥ 8%) | gate: observe / act / escalate |

Consequences for the response letter:
- The paper must name **two** consensus thresholds: τ = 0.65 for RER in the controlled studies and τ_AEI = 0.75 for the selective-AEI trigger. The draft justification of "τ = 0.65 ... balancing unnecessary Agentic invocations" mixes the two: Agentic invocation uses 0.75.
- R4-5: RCAEval CPU/disk faults got "No action" because the mapped CPU percentage stayed below AAF's absolute saturation thresholds (85/90%), so reliability severity was 0 — not because severity fell below δ_min (δ_min is a consensus-improvement threshold). Frozen artifact: CPU 12/13 and disk 12/13 with reliability_severity 0.0 although the CPU signal was mapped (`cpu_saturation_active = True`).
