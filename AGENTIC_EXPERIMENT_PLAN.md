# Hybrid Agentic AAF: Implementation and Experiment Plan

This document freezes the implementation direction before modifying the validated deterministic baseline.

## Research position

AAF is a hybrid governance architecture. Deterministic reasoning handles well-bounded operational conditions. Bounded Agentic AI is invoked selectively when evidence is incomplete, conflicting, or ambiguous. Agentic reasoning may identify missing evidence and invoke approved telemetry tools, but final governance authority remains in the deterministic cross-domain interaction, utility, and arbitration layers. The PM-facing LLM remains a bounded natural-language interaction and explanation layer.

## Configurations to compare

1. **Deterministic baseline**: current validated domain-agent logic + current deterministic cross-domain governance.
2. **Agentic-only baseline**: LLM domain agents with tool use and an LLM/agent recommendation, without deterministic governance arbitration.
3. **Hybrid AAF**: LLM domain agents + bounded tool use + agentic re-grounding + existing deterministic cross-domain governance and arbitration.

## Agent contract

Every domain agent must return a structured object with:

- `agent_type`
- `claim`
- `confidence`
- `evidence_ids`
- `evidence_summary`
- `proposed_action`
- `uncertainty`
- `needs_more_evidence`
- `requested_tools`
- `reasoning_trace_summary` (brief, non-chain-of-thought rationale suitable for audit)

Agents must not fabricate telemetry. Every evidence item used in a claim must reference an input evidence ID or an approved tool result ID.

## Approved tool families

- DevOps: deployment history, rollout status, restart history, CI/CD failures, configuration drift.
- SRE: latency/error/availability windows, saturation, restart trends, service health.
- FinOps: replica/resource footprint, utilization efficiency, cost/proxy indicators, scaling history.
- DevSecOps: vulnerability findings, policy violations, security scan results, release-gate status.

## Agentic re-grounding loop

1. Observe current evidence.
2. Produce a first structured assessment.
3. If uncertainty is above the configured boundary or required evidence is missing, request one or more approved tools.
4. Retrieve bounded evidence from the same incident context.
5. Re-run the domain assessment with the enriched evidence.
6. Accept the revised assessment only when it is evidence-backed and passes schema validation.
7. Pass structured agent outputs to the existing deterministic cross-domain governance layer.

## Experimental questions

- **RQ1 Cross-domain value**: Does explicit cross-domain governance outperform isolated/dominant-domain reasoning?
- **RQ2 Selective agency**: On ambiguous or incomplete evidence, does bounded Agentic AI improve evidence acquisition and action agreement relative to the deterministic baseline?
- **RQ3 Trustworthiness**: Does deterministic governance constrain unsupported, hallucinated, or policy-violating agent recommendations?
- **RQ4 PM interaction**: Does the bounded PM-facing LLM preserve authoritative decisions while enabling natural-language interpretation?

## Initial implementation gate

Do not start a full rerun immediately. First implement one end-to-end pilot covering:

- one clear-evidence case where no agentic escalation should occur;
- one cross-domain ambiguous case where agentic reasoning may be invoked;
- one incomplete-evidence case where the agent must request additional telemetry.

Only after schema validity, evidence traceability, decision preservation, and tool-call behavior are verified should the full benchmark rerun begin.

## Reuse policy

The validated deterministic algorithms, frozen benchmark cases, runtime adapters, statistical scripts, provenance mapping, and PM-facing LLM tests remain unchanged unless a new experiment explicitly requires an additive extension. The deterministic baseline is preserved as a frozen comparison and must not be silently modified while developing the Agentic layer.
