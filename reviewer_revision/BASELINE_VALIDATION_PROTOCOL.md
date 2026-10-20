# IEEE Access reviewer revision: baseline validation protocol (draft v0.1)

Status: **design/audit only; no new results claimed**. Original 32+16 cases, oracles, prompts, and historical runs are immutable. All new outputs must be stored separately with a new protocol identifier.

## Research question
Does bounded LLM-mediated evidence investigation improve governance-action agreement or evidence acquisition beyond a deterministic tool-fetch policy using the same evidence, tools, and deterministic final authority?

## Existing implementation audit
- `agentic_experiments/hybrid.py` implements `deterministic_assessment`, `selective_trigger`, `run_agentic_only`, and `run_hybrid`.
- `missing_fields` reads `_evidence[field].status == missing`.
- `_execute_requested_tool` is a *simulator*: it returns `case.tool_result` only when a requested tool matches `case.expected_tool`. These evaluator-only fields must **never** be supplied to a baseline selector. They are used only by the common simulated tool executor.
- Both frozen runners execute each Agentic configuration once per case, with model `gpt-5.6-luna` and reasoning effort `none`.

## Pre-register comparator arms before running
1. **D0 (existing)**: deterministic assessment, no tool retrieval.
2. **D1 (new)**: deterministic assessment; if explicit missing evidence is declared, select a tool from a fixed, documented `(domain, field) -> tool` mapping using only model-visible evidence and allowed tools; fetch once under the same executor contract; merge tool results; rerun deterministic assessment. No LLM. Define an explicit fallback for unmapped fields and no-result responses.
3. **H-selective (existing frozen)**: selective AEI and deterministic final authority.
4. **H-always (new)**: invoke the same bounded domain agents/tools for *every* case, merge acquired evidence, then recompute the final action with the same deterministic governance. Match the selective arm's prompt, model, and tools.
5. **A-strong (new exploratory)**: Agentic-only with governance policy available in prompt and a supported reasoning effort. Do not treat its performance as directly comparable to the frozen weaker baseline without noting the changed conditions.

## Fairness and leakage controls
- Tool selection in D1 must not access `id`, `stratum`, `focus`, `admissible_actions`, `expected_tool`, `tool_result`, or `oracle_domains`.
- Use the same simulated tool interface and budgets across D1, H-selective, and H-always; record unsuccessful calls and no-result responses.
- The tool executor may internally use hidden `expected_tool` to simulate results, but this is a **dataset limitation**, not evidence of independent live-tool validation.
- Lock the mapping, fallback, prompts, budgets, and output schema before executing new cases. Report any design choices influenced by inspecting the 48 original cases; use additional genuinely held-out cases for confirmatory inference.
- Do not silently combine original frozen and new exploratory results into a single confirmatory significance test.

## Required outputs
Per-case: dataset/version, stratum, baseline arm, action before/after, admissible-set match, invoked agents, requested and successful tools, LLM calls, input/output tokens, wall-clock latency, missing-field coverage, and errors. Aggregate by all four strata, with paired discordances and appropriate confidence intervals. Include negative results and unchanged failures.

## Immediate audit targets
- Recover and inspect the exact frozen raw records for AG-I01 and RP-I01.
- Establish a field-to-tool mapping from documented tool semantics **without** consulting expected tool labels.
- Inspect the four non-beneficial governance overrides.
- Check whether the frozen artifacts contain sufficient call counts and latency data to report original per-case operational overhead.
