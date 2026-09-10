# Selective Hybrid Agentic AAF — Experiments 2–4 and Prospective Replication

This directory contains the frozen selective-Agentic evaluation for Hybrid AAF.

**Core protocol rule:** **Uncertainty invokes agency; severity does not.**

The Agentic layer is bounded and advisory. After any Agentic evidence acquisition/re-grounding, deterministic cross-domain governance computes the authoritative final action.

## Study separation

The two datasets are deliberately separate and must remain separate in reporting:

| Study | Dataset | n | Status |
|---|---|---:|---|
| Primary Protocol-v2 study | `cases_v1.yaml` | 32 | frozen confirmatory evidence |
| Prospective replication | `cases_replication_v1.yaml` | 16 | frozen unseen-case replication |

Do **not** edit either dataset, its oracle/admissible actions, the selective trigger, prompts, tool contract, or governance policy to improve the reported outcomes. Any future exploratory work requires a new versioned dataset/protocol and must not be merged into these metrics.

## Experimental configurations

1. **Deterministic AAF** — frozen deterministic domain assessment and deterministic cross-domain governance.
2. **Agentic-only** — bounded LLM domain reasoning and approved read-only evidence tools without deterministic final cross-domain arbitration.
3. **Hybrid AAF** — deterministic first pass; Agentic reasoning only when the frozen uncertainty trigger fires; approved evidence acquisition/re-grounding; deterministic governance remains authoritative.

The model does not receive evaluator-only fields such as case ID, stratum, focus, oracle domains, admissible actions, expected Agentic behavior, expected tool, or hidden tool result as incident evidence. Tool results become visible only after an allowed tool is requested.

## Model configuration

Both manuscript-reported Agentic studies use:

```text
model: gpt-5.6-luna
reasoning_effort: none
Python reference CI: 3.11
```

Fresh API calls are stochastic replications. The manuscript-reported evidence is therefore tied to the frozen GitHub Actions executions and artifacts below.

## Primary Protocol-v2 study — 32 cases

Balanced strata: 8 CLEAR, 8 AMBIGUOUS, 8 INCOMPLETE, 8 MISLEADING.

Run locally as a fresh replication:

```bash
export OPENAI_API_KEY='...'
export OPENAI_MODEL='gpt-5.6-luna'
export OPENAI_REASONING_EFFORT='none'
python -m agentic_experiments.run_experiments_2_4
```

Generated outputs are written to `paper_results/agentic_experiments_2_4/` and contain:

- `metadata.json` — model/configuration and frozen dataset SHA-256
- `raw_outputs.json` — deterministic, raw Agentic-only, tool, and Hybrid records
- `case_results.csv` — one row per independent case
- `summary.json` — aggregate Experiment 2–4 metrics

### Frozen manuscript execution

- Workflow: `AAF Confirmatory Agentic Experiments 2-4`
- Run: `34428945881`
- Head SHA: `3f5913f0a7e8a935f28751a3d074de049ec34951`
- Artifact: `aaf-agentic-experiments-2-4-gpt-5-6-luna`
- Artifact ID: `10133945801`
- Artifact digest: `sha256:51775beb89de8d09b16f8e6449d900cdd0a42e3863c3f5d9f05fd8059939b583`
- Dataset SHA-256 recorded in artifact metadata: `98fd94278193d2c48fffb11ff0b76e96d8233585bf52e52570516f35911913b5`

### Frozen primary results

- Deterministic: `26/32 = 81.25%`
- Agentic-only: `10/32 = 31.25%`
- Hybrid: `27/32 = 84.375%`
- Hybrid invocation: `21/32 = 65.625%`
- CLEAR unnecessary invocation: `0/8`
- Always-on Agentic tokens: `91,441`
- Hybrid tokens: `44,879`
- Token avoidance: `50.92%`
- Governance overrides: `14`
- Beneficial overrides: `11`
- INCOMPLETE deterministic pre-re-grounding: `7/8`; Hybrid post-re-grounding: `8/8`
- One initially incorrect/unresolved INCOMPLETE case recovered after evidence acquisition/re-grounding
- MISLEADING Agentic-only: `2/8`; Hybrid: `7/8`

The primary study does **not** support a claim that Agentic AI generally outperforms deterministic AAF. The absolute governance-action gain is only `26/32 -> 27/32`.

## Prospective replication — 16 unseen cases

The original 32 cases were not modified after their outcomes were observed. A separate prospective dataset was then frozen with 4 CLEAR, 4 AMBIGUOUS, 4 INCOMPLETE, and 4 MISLEADING cases. Its INCOMPLETE cases exercise different evidence paths, including restart history, security policy gate, cost evidence, and SRE saturation.

Run locally as a fresh replication:

```bash
export OPENAI_API_KEY='...'
export OPENAI_MODEL='gpt-5.6-luna'
export OPENAI_REASONING_EFFORT='none'
python -m agentic_experiments.run_prospective_replication
```

Generated outputs are written to `paper_results/agentic_prospective_replication/` with the same four-file structure as the primary study.

### Frozen prospective execution

- Workflow: `AAF Prospective Agentic Replication`
- Successful run: `34432645600` (run #2)
- Head SHA: `2417f2f0062dd2a3541e9ce9904e1880fcf15b31`
- Artifact: `aaf-agentic-prospective-replication-gpt-5-6-luna`
- Artifact ID: `10135143965`
- Artifact digest: `sha256:a66f5aafb74027d15eabb8047230ed941ab84fa0c38e32233da091e64f2fa343`
- Dataset SHA-256 recorded in artifact metadata: `eb41fc780ead10e9ce422fe4c704565fd7ae1ccf84a4a48d27ceb52cd7ab898d`

### Frozen replication results

- Deterministic: `13/16 = 81.25%`
- Agentic-only: `7/16 = 43.75%`
- Hybrid: `14/16 = 87.5%`
- CLEAR invocation: `0/4`
- AMBIGUOUS invocation: `4/4`
- INCOMPLETE invocation: `4/4`
- MISLEADING invocation: `2/4`
- Overall Hybrid invocation: `10/16 = 62.5%`
- Always-on Agentic tokens: `47,351`
- Hybrid tokens: `18,682`
- Token avoidance: `60.5%`
- Governance overrides: `8`
- Beneficial overrides: `7`
- INCOMPLETE deterministic: `3/4`; Hybrid: `4/4`
- Expected tool selected by Hybrid: `4/4`

`RP-A04` remains incorrect under both deterministic and Hybrid AAF. It is preserved as a genuine boundary/failure case.

## Combined 48-case interpretation

Across the primary and prospective studies:

- Deterministic AAF: `39/48 = 81.25%`
- Agentic-only: `17/48 = 35.42%`
- Hybrid AAF: `41/48 = 85.42%`

The scientifically appropriate interpretation is architectural rather than a generic LLM-accuracy claim: deterministic cross-domain governance is already strong on bounded cases; selective Agentic reasoning provides targeted value when missing evidence must be acquired and re-grounded; deterministic governance prevents many incorrect Agentic proposals from becoming authoritative actions; and selective invocation reduces LLM usage substantially relative to always-on Agentic reasoning.

## Artifact preservation policy

The manuscript-reported raw outputs are retained as immutable GitHub Actions artifacts and identified above by run, commit, artifact ID and digest. Generated execution directories are intentionally not treated as source code. Deterministic rescoring/analysis should use the frozen artifact outputs; a new API execution is a stochastic replication, not a silent replacement for the reported run.