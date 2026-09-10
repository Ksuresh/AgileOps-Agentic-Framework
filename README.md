# AgileOps Agentic Framework (AAF)

Reproducibility repository for the Hybrid AAF manuscript and its deterministic, Agentic, replication, external-telemetry, and bounded-PM-interface evidence.

AAF is a **hybrid deterministic + selectively Agentic AI decision-support framework for Agile–DevOps project governance**. Deterministic cross-domain reasoning remains authoritative for final governance actions. Bounded Agentic reasoning is invoked only when decision-relevant evidence is uncertain, incomplete, ambiguous, or requires additional evidence acquisition. A separate bounded LLM interface allows Project Managers to interrogate and understand an already-governed recommendation.

> **Core design principle:** **Uncertainty invokes agency; severity does not.**
>
> The Agentic/LLM layer does **not** autonomously determine the authoritative governance action.

> **Start here to reproduce the paper:** [REPRODUCE_CURRENT_PAPER.md](REPRODUCE_CURRENT_PAPER.md)  
> **Reviewer-facing experiment/results map:** [paper_results/README.md](paper_results/README.md)  
> **Manuscript-to-run provenance:** [paper_results/MANUSCRIPT_RESULT_MAP.md](paper_results/MANUSCRIPT_RESULT_MAP.md)

## Architecture at a glance

```text
Operational telemetry / evidence
        |
        v
Deterministic domain assessment
        |
        v
Uncertainty / ambiguity gate
        |
        +---------------- sufficient evidence ----------------+
        |                                                     |
        | uncertain / incomplete                              |
        v                                                     |
Selective bounded Agentic domain reasoning                    |
        |                                                     |
        v                                                     |
Approved read-only tools / evidence acquisition                |
        |                                                     |
        v                                                     |
Evidence re-grounding                                          |
        |                                                     |
        +--------------------------+--------------------------+
                                   |
                                   v
Deterministic cross-domain interaction / readiness / utility
                                   |
                                   v
Authoritative governance action
                                   |
                                   v
Optional bounded PM-facing LLM explanation / interaction
```

The final arbitration order is:

```text
hard governance override -> specific causal interaction -> generic accumulation
```

A governance evaluation may be triggered by a deployment event, monitoring/security event, scheduled checkpoint, or explicit PM request to evaluate current evidence. A PM request is an instruction to evaluate; it is not operational evidence.

## Three experimental configurations

The Agentic studies compare three deliberately different configurations:

1. **Deterministic AAF** — frozen deterministic domain assessment plus deterministic cross-domain governance.
2. **Agentic-only** — bounded LLM domain reasoning and approved evidence tools without deterministic final cross-domain arbitration.
3. **Hybrid AAF** — deterministic first pass; selective Agentic reasoning only when the frozen uncertainty trigger fires; approved evidence acquisition and re-grounding; deterministic governance remains authoritative.

The Hybrid architecture is not designed to maximize LLM invocation. Avoiding unnecessary Agentic calls on clear evidence is an intended outcome.

## Validation strategy

The manuscript uses complementary validation layers that answer different research questions and must not be collapsed into a single accuracy number:

| Layer | Purpose |
|---|---|
| Controlled cross-domain experiments | isolate arbitration, interaction and ablation effects under frozen oracles |
| Live benchmark validation | test measured evidence from Sock Shop and Google Online Boutique under controlled interventions |
| Primary selective-Agentic study | compare Deterministic, Agentic-only and Hybrid AAF on 32 frozen CLEAR/AMBIGUOUS/INCOMPLETE/MISLEADING cases |
| Prospective Agentic replication | test the frozen Protocol-v2 architecture on 16 unseen cases without modifying the original 32 cases |
| Independent external-fault validation | test label-blind AAF behavior on RCAEval RE2-TrainTicket recorded fault evidence |
| Bounded PM-facing LLM faithfulness | test decision preservation, evidence faithfulness and override resistance downstream of governance |

RCAEval does **not** provide project-governance action ground truth. It is used to demonstrate external telemetry ingestion and interpretation, not governance-action accuracy.

## Key Agentic evidence

### Primary frozen study — 32 cases

- Deterministic AAF: **26/32 = 81.25%**
- Agentic-only: **10/32 = 31.25%**
- Hybrid AAF: **27/32 = 84.375%**
- Hybrid invocation: **21/32 = 65.625%**
- CLEAR unnecessary invocation: **0/8**
- Always-on Agentic tokens: **91,441**
- Hybrid tokens: **44,879**
- Token avoidance: **50.92%**
- Governance overrides: **14**, of which **11** were beneficial

Frozen Protocol-v2 execution: workflow **AAF Confirmatory Agentic Experiments 2-4**, run `34428945881`, head SHA `3f5913f0a7e8a935f28751a3d074de049ec34951`, artifact ID `10133945801`, artifact digest `sha256:51775beb89de8d09b16f8e6449d900cdd0a42e3863c3f5d9f05fd8059939b583`.

### Prospective replication — 16 unseen cases

- Deterministic AAF: **13/16 = 81.25%**
- Agentic-only: **7/16 = 43.75%**
- Hybrid AAF: **14/16 = 87.5%**
- Hybrid invocation: **10/16 = 62.5%**
- CLEAR unnecessary invocation: **0/4**
- Always-on Agentic tokens: **47,351**
- Hybrid tokens: **18,682**
- Token avoidance: **60.5%**
- Governance overrides: **8**, of which **7** were beneficial
- INCOMPLETE: deterministic **3/4** -> Hybrid **4/4**; expected tool selected **4/4** in Hybrid

Frozen prospective execution: workflow **AAF Prospective Agentic Replication**, run `34432645600`, head SHA `2417f2f0062dd2a3541e9ce9904e1880fcf15b31`, artifact ID `10135143965`, artifact digest `sha256:05abd88dd2359d3d7a14b061fc42d74dec3c489a8f638488c35339823017428e`.

`RP-A04` remains incorrect under both Deterministic and Hybrid AAF and is retained as a genuine boundary/failure case.

### Combined architectural evidence — 48 cases

- Deterministic AAF: **39/48 = 81.25%**
- Agentic-only: **17/48 = 35.42%**
- Hybrid AAF: **41/48 = 85.42%**

These results are **not** interpreted as evidence that Agentic AI generally outperforms deterministic AAF. The intended contribution is architectural: deterministic cross-domain governance already performs strongly on bounded cases; selective Agentic reasoning adds targeted value when evidence acquisition/re-grounding is needed; deterministic authority prevents many incorrect Agentic proposals from becoming governance actions; selective invocation materially reduces LLM usage relative to an always-on Agentic design.

## Quick start

```bash
git clone https://github.com/Ksuresh/AgileOps-Agentic-Framework.git
cd AgileOps-Agentic-Framework
git checkout experiment/hybrid-agentic-aaf-v2
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
export PYTHONPATH="$PWD"         # PowerShell: $env:PYTHONPATH=(Get-Location)
python -m pytest -q
```

Python 3.11 is the reference CI version. Runtime studies additionally require Docker/Docker Compose (Sock Shop) or Docker + kubectl + kind (Online Boutique). Fresh Agentic runs require an OpenAI API key and are stochastic replications; the manuscript-reported executions are identified by immutable run/artifact provenance.

## Repository map

```text
agents/                 deterministic domain evidence modules
agentic_experiments/    frozen selective-Agentic protocols, datasets, runtime and replication
benchmark/              frozen controlled-experiment oracle definitions
config/                 framework/experiment configuration
evaluation/             comparison, statistical and sensitivity analyses
evidence/               structured evidence schema
experiments/            controlled/held-out and learned-baseline experiments
external_validation/    RCAEval external evidence adapter/protocol
llm_evaluation/         bounded PM-facing LLM interface and faithfulness evaluation
orchestrator/           materiality, interaction, consensus, arbitration and utility logic
runtime_validation/     live benchmark interventions, adapters and evaluators
scenario_generator/     deterministic controlled scenario generation
tests/                  regression and policy tests
paper_results/          reviewer-facing results/provenance index
.github/workflows/      executable CI/reproduction workflows
```

## Agentic evidence contract and guardrails

When invoked, a domain component follows a bounded observe/reason/evidence-acquisition/re-grounding loop and returns a structured record containing `agent_type`, `claim`, `confidence`, `evidence_ids`, `proposed_action`, `uncertainty`, `needs_more_evidence`, `requested_tools`, and `rationale_summary`.

`rationale_summary` is a concise auditable explanation, not private chain-of-thought. Tool access is read-only and allow-listed. The runtime validates action choices, tool requests, output schema and evidence references. Evaluator-only fields such as case ID, stratum, oracle action, expected tool and hidden tool result are not supplied as model-visible incident evidence.

## Bounded PM-facing LLM interface

The downstream PM-facing LLM is constrained to interpret/verbalize supplied structured AAF outputs. It does not become the governance authority and must not invent telemetry, causal links, incidents, risks or business impact.

The frozen 100-prompt GPT-5.6 Luna faithfulness study recorded intent mapping **91/100**, authoritative-action preservation **100/100**, adversarial authority resistance **25/25**, unsupported-evidence lure rejection **25/25**, and **0/100** unsupported numeric claims.

We do **not** claim that an LLM cannot hallucinate. The architecture is designed so that an unsupported LLM conclusion does not automatically become a governance decision.

## Reproducibility principles

- Evidence is collected before AAF inference; intervention/oracle labels are not supplied to AAF.
- Frozen manifests, held-out snapshots, Agentic datasets and prompt/protocol definitions are not tuned after inspecting outcomes.
- The original 32-case Agentic study and the later 16-case prospective replication remain separate studies.
- Benchmark source revisions are pinned for confirmatory/prospective runtime studies.
- Raw execution evidence and workflow artifacts are distinguished from later reruns.
- Resource scaling is called a resource-footprint proxy where monetary billing evidence is unavailable.
- LLM raw behavior is reported separately from deterministic system-level authority enforcement.
- Development iterations remain in Git history unless they materially affect scientific interpretation.

## Historical provenance

The former pre-paper main is preserved at `archive/pre-paper-2026-main`. Historical experiment branches and GitHub Actions runs preserve exact execution states. Reviewer-facing documentation must use the frozen provenance identifiers in `paper_results/` rather than infer results from historical directory names.

## License

See `LICENSE`.