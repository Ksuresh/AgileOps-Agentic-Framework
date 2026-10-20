# Stage 2: Agentic incremental-value evaluation (preregistration draft v0.1)

Status: **design only, not executed**. The D1 result (41/48) was inspected before drafting this protocol. Therefore all 48 historical cases are **development/exploratory data** and must never be used to claim confirmatory Agentic superiority.

## Empirical motivation
The 2026-10-20 GitHub Actions run 39405009632 executed D1 successfully: D0 39/48, D1 41/48. D1 recovered AG-I01 and RP-I01 via get_restart_history. Original Hybrid AAF was 41/48. Thus the original aggregate gain is explained by deterministic retrieval on this dataset. The new study must test incremental value beyond D1, not repeat the old D0 contrast.

## Research questions
RQ1: With the same evidence and available tools, does bounded LLM-mediated investigation improve correct tool selection or actionable evidence acquisition compared with deterministic retrieval?
RQ2: Conditional on identical retrieved evidence and identical deterministic governance, does Agentic investigation improve final action agreement?
RQ3: What additional latency, token cost, tool calls, and error risk does Agentic investigation impose?

## Primary confirmatory endpoint
Paired per-case **final governed-action correctness** against an independently adjudicated admissible action set, H-selective versus D1, under the same tool-call budget and executor. Report the paired difference in proportions with a cluster-bootstrap 95% CI (cluster by incident family), exact McNemar discordance counts and p-value (descriptive unless predeclared power supports confirmatory testing). Correctness means selected_action belongs to the pre-frozen admissible set; no post-hoc oracle changes.

## Secondary endpoints
- Correct useful tool selected at or before budget; successful evidence acquisition; fraction of missing decision-relevant fields resolved.
- Unsafe-action rate and unsupported evidence-reference rate, adjudicated blind.
- LLM calls, input/output tokens, total requested and successful tool calls, wall-clock latency, monetary API cost at recorded rate.
- Per-stratum performance and paired discordances, including cases where H loses to D1.
- Optional explanation quality via blinded rubric and inter-rater agreement; do not infer PM adoption or business outcomes without a human study.

## Comparator arms and fairness
D0: deterministic without retrieval.
D1: frozen fixed field-to-tool map and deterministic authority.
D2: **strong deterministic retrieval** using documented tool descriptions, configurable multiple calls, explicit fallback on no-result, and an evidence-coverage heuristic; lock before held-out data.
H-selective: bounded LLM investigation only when a prespecified uncertainty trigger fires, then identical deterministic final authority.
H-always: same LLM agent and tools on all cases, then identical deterministic final authority.
A-strong: policy-informed Agentic-only with a supported reasoning effort; explicitly exploratory if settings differ.
All arms see the same case-visible evidence, allowed tools, tool descriptions, budgets, and response availability. Separate runs with one-call and multi-call budgets. Keep model name, snapshot, prompt SHA, temperature (where supported), reasoning effort, seeds (where supported), retries and rate-limit policy in a machine-readable manifest.

## Leakage barriers
Selectors must not receive case ID, stratum, focus, expected_tool, tool_result, oracle_domains, admissible_actions, or any reference action. The executor and scorer operate behind separate interfaces. Critically, replace the historical expected_tool-gated simulator for the held-out experiment with a **tool response table** that defines independent responses for every offered tool, including plausible distractors and no-result cases. Otherwise the simulator itself rewards the one expected tool. Tool catalog and deterministic mappings must be fixed without examining held-out outcomes.

## Held-out data and independent oracle
Recruit two independent domain reviewers to adjudicate the admissible action set and evidence requirements from a structured incident record, blinded to algorithm predictions. Resolve disagreement via a third adjudicator, retaining original ratings and rationale. Build genuinely new incident families from de-identified real incidents where permission exists, plus clearly labeled synthetic stress tests; no copies or parameter tweaks of the 48 development cases. Stratify by explicit missingness, implicit missingness, competing evidence, conflicting telemetry, misleading evidence, unavailable tools, stale evidence, and tool failures. Freeze cases, oracle, tool-response table, and split manifests with hashes **before** first held-out run.

## Sample size and inference
Do not declare a sample size based on arbitrary convenience. Run a paired power/sensitivity calculation using a plausible discordance rate and smallest practically meaningful gain, accounting for family clustering. Freeze the resulting target N before evaluating the held-out set. If resources limit N, label the study exploratory and report uncertainty rather than significance claims. Use one predeclared primary contrast and adjust or label other contrasts exploratory.

## Execution and audit gates
1. Write leakage and parity unit tests (including wrong-tool, absent-tool, and multi-call behavior).
2. Implement D2, H-always, and A-strong with common tool executor and standardized per-case audit logs.
3. Create and hash a held-out corpus and independent oracle; record annotator agreement.
4. Freeze manifests and analysis scripts; run all arms under the same budget.
5. Reproduce all tables directly from raw outputs; publish discordant cases and negative results.
6. Run robustness checks for prompt/model variation, retries, response latency, missing metadata, and tool failures.
7. Only then revise claims, manuscript, and point-by-point reviewer responses.

## Stop/go rule
If H-selective does not outperform D1/D2 on the primary endpoint, **do not claim Agentic-driven action-accuracy improvement**. Consider an evidence-investigation, explanation, or governance-safety contribution only if its corresponding independently validated endpoint supports it. Null results remain part of the revision.
