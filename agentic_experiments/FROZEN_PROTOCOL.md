# Frozen Protocol: Selective Hybrid Agentic AAF Experiments 2-4

Status: **frozen before any LLM execution on this dataset**.

## Purpose

Experiments 2-4 test the incremental value and risk-control properties of selective Agentic AI. They do not re-test whether cross-domain governance matters; that is established by the existing deterministic controlled and runtime studies.

## Experimental configurations

1. **Deterministic AAF** - existing frozen domain rules plus deterministic cross-domain governance.
2. **Agentic-only** - bounded LLM domain reasoning and approved evidence tools, without deterministic final cross-domain arbitration.
3. **Hybrid Agentic AAF** - deterministic first pass; bounded LLM agentic reasoning only when a pre-specified uncertainty trigger fires; approved evidence-tool retrieval; existing deterministic cross-domain governance remains authoritative.

The existing deterministic code is a frozen baseline and must not be altered to improve results on this dataset.

## Dataset design

The primary Agentic evaluation dataset contains **32 independent evidence cases**, divided equally into four pre-specified strata (8 each):

- **CLEAR**: evidence is deliberately well separated from deterministic thresholds. Selective AAF should normally avoid Agentic invocation.
- **AMBIGUOUS**: two materially plausible interpretations/actions coexist or evidence lies near a governance boundary. Selective Agentic reasoning may be useful even before extra evidence is requested.
- **INCOMPLETE**: one decision-relevant evidence item is explicitly withheld and marked `missing`. Each case has one pre-specified evidence tool whose result resolves the intended information gap. These cases test Agentic RAR.
- **MISLEADING**: a salient but non-decisive signal is present alongside stronger evidence. These cases test whether an LLM agent overreacts and whether deterministic governance constrains unsupported/unsafe recommendations.

The cases are balanced across the four operational perspectives (DevOps, SRE, FinOps, DevSecOps) and the cross-domain interaction types used by the validated governance layer: deployment-reliability, deployment-security, reliability-resource, and multi-domain accumulation. The dataset is synthetic-by-design so missing evidence, misleading evidence, tool availability, and oracle actions can be pre-specified without leaking benchmark intervention labels to the model.

## Parameter-selection rule

Case values are generated from the already frozen deterministic evidence schema and thresholds rather than selected after observing LLM outputs.

- CLEAR cases use values materially away from decision thresholds.
- AMBIGUOUS cases use values near existing thresholds or paired cross-domain evidence that supports competing local interpretations.
- INCOMPLETE cases are paired derivatives of underlying evidence patterns with exactly one decisive evidence field hidden at first observation.
- MISLEADING cases add a plausible but non-decisive signal; the oracle is determined by the full evidence semantics, not by the most salient single signal.

No case may be added, removed, relabeled, or have its oracle/tool target changed after the first LLM run. Any later exploratory cases must be stored in a separately versioned exploratory dataset and excluded from confirmatory metrics.

## Information available to an LLM agent

The model receives only:

- incident-scoped evidence values and evidence IDs;
- evidence status (`measured`, `proxy`, `missing`, `not_applicable`);
- its domain role and allowed tool catalogue;
- the fixed structured-output schema.

The model must **not** receive scenario IDs that encode semantics, stratum names, oracle domains, admissible actions, expected tool calls, or benchmark intervention labels.

## Agentic trigger for Hybrid AAF

Agentic reasoning is eligible only if one or more of the following pre-specified conditions is true:

1. a decision-relevant field is explicitly `missing`;
2. at least two materially relevant domains support competing candidate actions;
3. deterministic decision readiness is below the frozen readiness threshold;
4. a required temporal relationship cannot be established from available evidence;
5. a high-impact proposed action lacks sufficient evidence provenance.

The trigger must not use oracle actions or whether the deterministic baseline is correct.

## Approved tool families

Tool calls are read-only and incident-scoped.

- DevOps: deployment history, rollout status, restart history, pipeline/configuration evidence.
- SRE: latency/error/availability history, saturation history, restart/service-health history.
- FinOps: replica/resource history, utilization-efficiency evidence, cost/resource proxy history.
- DevSecOps: vulnerability details, policy-gate status, IAM/compliance evidence.

For the 8 INCOMPLETE cases, `cases_v1.json` pre-specifies the only tool result required to resolve the deliberately withheld evidence. The LLM sees available tool names but not `expected_tool`.

## Structured agent output

Each LLM domain agent must return JSON containing:

- `agent_type`
- `claim`
- `confidence` in [0,1]
- `evidence_ids`
- `proposed_action`
- `uncertainty` in [0,1]
- `needs_more_evidence`
- `requested_tools`
- `rationale_summary`

`rationale_summary` is a short audit explanation, not private chain-of-thought. Claims and actions must be supported by referenced evidence/tool-result IDs.

## Experiment 2 - selective agency and efficiency

Question: **When evidence is sufficient, can Hybrid AAF preserve deterministic decisions while avoiding unnecessary LLM calls?**

Primary metrics:

- governance-action agreement;
- Agentic invocation rate by stratum;
- unnecessary invocation rate on CLEAR cases;
- total/input/output tokens;
- token avoidance relative to always-on Agentic reasoning;
- latency and tool-call count.

A low Agentic invocation rate on CLEAR cases is treated as a positive efficiency outcome, not a failure to use AI.

## Experiment 3 - Agentic RAR under incomplete evidence

Question: **Can an Agentic domain reasoner identify a missing information need, select the appropriate evidence tool, and improve the final governed decision?**

Primary metrics on the 8 INCOMPLETE cases:

- missing-evidence detection;
- expected-tool selection accuracy;
- evidence-acquisition success;
- evidence-ID faithfulness;
- pre/post re-grounding action agreement;
- re-grounding success among initially unresolved/incorrect cases.

## Experiment 4 - trustworthiness under misleading evidence

Question: **Does deterministic governance reduce the effect of unsupported or over-reactive LLM recommendations?**

Primary metrics on the 8 MISLEADING cases and a fixed adversarial prompt variant per case:

- unsupported-claim rate;
- unsupported-action rate;
- valid evidence-reference rate;
- agentic-only action agreement;
- Hybrid AAF action agreement;
- deterministic governance override rate;
- beneficial override rate (override changes an incorrect agent proposal to an admissible governed action).

## Statistical analysis

The unit of analysis is the independent case, not repeated LLM calls. The primary comparison is paired because configurations receive the same case evidence. Report exact counts/percentages and Wilson 95% confidence intervals. For paired binary action-agreement comparisons, use exact two-sided McNemar tests on discordant cases. Token/latency measures are descriptive and reported with median and interquartile range in addition to the mean where useful.

The 32-case study is deliberately focused rather than presented as a population estimate. Effect interpretation must emphasize the pre-specified strata and paired within-case behavior. If a larger replication is later run, it is a separate replication and must not silently replace this frozen study.

## Model and reproducibility policy

Use one fixed model/configuration for the confirmatory run. Record model identifier, API endpoint family, reasoning setting, prompt version/hash, complete model-visible inputs, tool calls/results, raw model outputs, token usage, timestamps, validator outcomes, and final governed outputs.

Fresh LLM calls are stochastic replications; the paper's reported analysis must also be reproducible by deterministic rescoring of frozen raw outputs.
