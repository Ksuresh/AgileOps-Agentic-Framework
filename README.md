# AgileOps Agent Framework (AAF)

Reproducibility repository for the manuscript:

**Cross-Domain Evidence Integration for Project Governance in Agile–DevOps: An Explainable Multi-Agent Decision-Support Framework**

AAF is a **deterministic, evidence-driven multi-agent decision-support framework** for integrating DevOps, Site Reliability Engineering (SRE), FinOps and DevSecOps evidence into bounded project-governance recommendations. The optional LLM interface can interpret supported PM questions and verbalize an already-computed structured decision; it is outside the authoritative decision path and cannot create operational evidence.

> **Start here to reproduce the paper:** [REPRODUCE_CURRENT_PAPER.md](REPRODUCE_CURRENT_PAPER.md)  
> **Reviewer-facing experiment/results map:** [paper_results/README.md](paper_results/README.md)  
> **Manuscript-to-run provenance:** [MANUSCRIPT_RESULT_MAP.md](MANUSCRIPT_RESULT_MAP.md)

## Architecture at a glance

```text
Governance evaluation trigger
        |
        v
Structured evidence collection
        |
        v
DevOps | SRE | FinOps | DevSecOps modules
        |
        v
Materiality gate
        |
        v
Cross-domain interaction detection
        |
        v
Decision-readiness consensus
        |
        v
Bounded evidence re-grounding
        |
        v
Evidence-constrained utility / action selection
        |
        v
Structured AAF decision record
        |
        +----> optional bounded PM LLM interface
```

A governance evaluation may be triggered by a deployment event, monitoring/security event, scheduled checkpoint, or explicit PM request to evaluate current evidence. A PM request is an instruction to evaluate; it is not operational evidence.

## Validation strategy

The manuscript uses four complementary validation layers:

| Layer | Purpose |
|---|---|
| Controlled cross-domain experiments | isolate arbitration, interaction and ablation effects under frozen oracles |
| Live benchmark validation | test measured evidence from Sock Shop and Google Online Boutique under controlled interventions |
| Independent external-fault validation | test label-blind AAF behavior on RCAEval RE2-TrainTicket recorded fault evidence |
| Bounded-LLM faithfulness | test decision preservation, evidence faithfulness and override resistance of the optional PM interface |

The studies intentionally answer different questions; RCAEval fault labels, for example, are not treated as governance-action labels.

## Quick start

```bash
git clone https://github.com/Ksuresh/AgileOps-Agentic-Framework.git
cd AgileOps-Agentic-Framework
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
export PYTHONPATH="$PWD"         # PowerShell: $env:PYTHONPATH=(Get-Location)
python -m pytest -q
```

Python 3.11 is the reference CI version. Runtime studies additionally require Docker/Docker Compose (Sock Shop) or Docker + kubectl + kind (Online Boutique).

## Repository map

```text
agents/                 deterministic domain evidence modules
benchmark/              frozen controlled-experiment oracle definitions
config/                 framework/experiment configuration
evaluation/             comparison, statistical and sensitivity analyses
evidence/               structured evidence schema
experiments/            controlled/held-out and learned-baseline experiments
external_validation/    independent external-evidence adapters/protocols (when consolidated)
llm_evaluation/         bounded PM interface and faithfulness evaluation
orchestrator/           materiality, interaction, consensus, arbitration and utility logic
runtime_validation/     live benchmark interventions, adapters and evaluators
scenario_generator/     deterministic controlled scenario generation
tests/                  regression and policy tests
paper_results/          reviewer-facing results/provenance index
.github/workflows/      executable CI/reproduction workflows
```

## Evidence sources

AAF does **not** depend on one conventional ML dataset. Controlled evidence is deterministically generated from frozen configurations/seeds. Sock Shop and Online Boutique are open-source benchmark applications used to produce measured runtime evidence under pre-specified interventions. RCAEval RE2-TrainTicket is used as an independent recorded-fault evidence source. The learned baselines use generated calibration scenarios and a frozen held-out HRT feature snapshot.

None of these benchmark studies is presented as customer-production data.

## Current reproducibility status

The repository contains executable controlled, Sock Shop, Online Boutique, learned-baseline and bounded-LLM paths, with historical workflow/artifact provenance documented under `paper_results/` and `MANUSCRIPT_RESULT_MAP.md`. External RCAEval validation has a frozen successful held-out execution and is being consolidated from its provenance branch into the reviewer-ready tree.

One limitation is kept explicit: the current unified Online Boutique workflow executes **OB-01--OB-08**. It must not be described as reproducing a historical 16-case result until that larger execution path is itself consolidated and verified.

## Bounded LLM interface

The LLM has only two permitted roles: map a PM question to a supported query intent and verbalize the supplied structured AAF decision record. It must not change the selected action or invent telemetry, causal links, incidents, risks or business impact.

The frozen 100-prompt GPT-5.6 Luna faithfulness study achieved 100/100 raw and system-level authoritative-action preservation. Contextual post-hoc rescoring of the same frozen outputs found 25/25 adversarial override attempts resisted and 25/25 unsupported-evidence prompts rejected. See [`paper_results/llm_faithfulness/README.md`](paper_results/llm_faithfulness/README.md) for the exact provenance and interpretation boundary.

## Reproducibility principles

- Evidence is collected before AAF inference; intervention/oracle labels are not supplied to AAF.
- Frozen manifests, held-out snapshots and prompt sets are not tuned after inspecting outcomes.
- Benchmark source revisions are pinned for confirmatory/prospective studies.
- Raw execution evidence and workflow artifacts are distinguished from later reruns.
- Resource scaling is called a resource-footprint proxy where monetary billing evidence is unavailable.
- LLM raw behavior is reported separately from deterministic system-level authority enforcement.
- Historical branches are retained for provenance while reviewer-facing execution paths are consolidated.

## Historical provenance

The former pre-paper main is preserved at `archive/pre-paper-2026-main`. Historical experiment branches and GitHub Actions runs preserve exact execution states. Reviewers should use the current reproduction guide and the provenance identifiers in `paper_results/` rather than guessing from historical directory names.

## License

See `LICENSE`.
