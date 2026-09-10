# Hybrid Agentic AAF: Frozen Implementation and Experiment Record

This document records the implementation direction used for the selective-Agentic extension of AAF. The manuscript-facing terminology is **Agentic Evidence Investigation (AEI)** for LLM-mediated investigation and **evidence re-grounding/reassessment** for the deterministic processing that follows newly acquired evidence.

> Historical note: earlier development notes used terms such as “agentic re-grounding.” Those labels are retained only in Git history; the frozen datasets, prompts, triggers, outputs, and reported results are not modified by this documentation cleanup.

## Research position

AAF is a hybrid governance architecture. Deterministic reasoning handles well-bounded operational conditions. Bounded Agentic AI is invoked selectively when decision-relevant evidence is incomplete, conflicting, or ambiguous. Agentic reasoning may identify missing evidence and invoke approved read-only telemetry tools, but final governance authority remains in deterministic cross-domain interaction, utility, and arbitration layers. The PM-facing LLM remains a bounded natural-language interaction and explanation layer downstream of governance.

**Core design principle:** **Uncertainty invokes agency; severity does not.**

## Frozen configurations

1. **Deterministic AAF** — frozen deterministic domain assessment plus deterministic cross-domain governance.
2. **Agentic-only** — bounded LLM domain reasoning and approved evidence tools without deterministic final cross-domain arbitration.
3. **Hybrid AAF** — deterministic first pass; selective AEI only when the frozen uncertainty trigger fires; approved evidence acquisition followed by deterministic evidence re-grounding/reassessment; deterministic governance remains authoritative.

## Frozen Agent contract

Every invoked domain agent returns a structured record containing:

- `agent_type`
- `claim`
- `confidence`
- `evidence_ids`
- `proposed_action`
- `uncertainty`
- `needs_more_evidence`
- `requested_tools`
- `rationale_summary` — brief auditable rationale, not private chain-of-thought

Agents must not fabricate telemetry. Evidence references must resolve to model-visible input evidence IDs or approved tool-result IDs. The runtime validates schema, actions, requested tools, and evidence references.

## Approved read-only tool families

- DevOps: deployment/pipeline/restart history.
- SRE: latency, error-rate, saturation and restart history.
- FinOps: replica and cost-proxy history.
- DevSecOps: policy-gate status and vulnerability details.

The executable allow-list in `agentic_experiments/agentic_runtime.py` is authoritative.

## AEI and evidence re-grounding flow

1. Assess current evidence deterministically.
2. Apply the frozen uncertainty/ambiguity trigger.
3. When triggered, invoke bounded AEI for the relevant domain(s).
4. If decision-relevant evidence is missing, AEI may request only approved read-only tools.
5. Add returned evidence to the incident evidence packet with provenance.
6. Re-ground/reassess the evidence.
7. Run deterministic cross-domain interaction, readiness/utility, and arbitration.
8. Preserve the deterministic governance result as the authoritative action.
9. Optionally expose that already-governed record to the bounded PM-facing LLM.

## Validation questions

The final manuscript organizes the validation around four research questions:

- **RQ1 — Cross-domain governance:** Does explicit cross-domain governance outperform reduced isolated/dominant-domain reasoning under controlled and runtime evidence?
- **RQ2 — Prospective arbitration:** Does the finalized arbitration mechanism preserve the intended cross-domain behavior on a separately prospective runtime block?
- **RQ3 — Selective agency:** Does bounded AEI provide targeted value under ambiguous/incomplete evidence while avoiding unnecessary LLM invocation and preserving deterministic authority?
- **RQ4 — External and conversational boundary:** Can AAF ingest independent recorded telemetry, and can the downstream PM-facing LLM preserve authoritative decisions and evidence boundaries?

The authoritative manuscript-to-execution mapping is `paper_results/MANUSCRIPT_RESULT_MAP.md`.

## Frozen-study policy

The primary 32-case Protocol-v2 study and the later 16-case prospective replication are separate frozen studies. Do not edit either dataset, admissible actions/oracles, selective trigger, prompts, tool contract, or governance policy to improve reported outcomes. Fresh LLM executions are stochastic replications and must not silently replace the manuscript-reported GitHub Actions artifacts.

## Interpretation boundaries

- The selective-Agentic evidence does not establish that Agentic AI generally outperforms deterministic AAF.
- AEI is selectively invoked to investigate uncertainty or acquire missing evidence; it is not the authoritative governance layer.
- Evidence re-grounding/reassessment after acquisition is distinct from the LLM-mediated AEI step.
- The PM-facing LLM is downstream of governance and cannot autonomously convert an unsupported conclusion into the authoritative action.
- RCAEval supplies independent recorded telemetry but not project-governance action ground truth.

For current execution instructions use `REPRODUCE_CURRENT_PAPER.md`; for reviewer-facing results use `paper_results/README.md`; for exact frozen provenance use `paper_results/MANUSCRIPT_RESULT_MAP.md`.