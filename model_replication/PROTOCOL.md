# Study 10: Second-model replication with an open-weight LLM (frozen before any run)

## Purpose
Reviewers asked whether the selective-investigation results depend on one LLM. AAF's central design claim is that
deterministic governance holds final decision authority, so the final action should depend little on which LLM is used,
whereas configurations in which the LLM decides (B5, B6) should depend on it. This study tests that claim with a second,
open-weight model family.

## Model and runtime
- Model: `qwen3:8b` served by Ollama on GitHub-hosted CPU runners; the model digest and Ollama version are recorded per shard
  (`results/<model>/<label>/env_shard_*.txt`).
- Sampling: temperature 0.7, context 8192, max 3072 output tokens; JSON output mode. No seed (runs are independent samples).
- "Reasoning none" maps to `think=false`; B6's "reasoning medium" maps to `think=true`.
- Code paths, prompts, case files and arms are those of Studies 6 and 7, unchanged; only `_call` is replaced (`run_qwen.py`).

## Cases and arms
- Cases: the 48 Study 6 cases (`cases_v1.yaml`, `cases_replication_v1.yaml`) and the 48 Study 7 cases (`cases_new_v1.yaml`).
- Arms: AAF (3 runs), B5 original Agentic-only (3 runs), B6 policy-aware Agentic-only with thinking (2 runs). B3 is derived
  offline from every B5 run by deterministic re-decision; B1, B2 variants need no LLM and are unchanged.
- Keys: Study 6 cases use the authors' key; Study 7 cases are scored on both the practitioner key (primary) and the authors'
  sealed key, as in Study 7.

## Hypotheses and analysis (fixed now)
- **H10.1 (primary): model independence of the final action.** For each case, compare the majority action under GPT
  (frozen Study 6/7 runs) with the majority action under Qwen. Prediction: the proportion of cases with the same action is
  higher for AAF than for B5 and for B6. Test: exact McNemar on paired per-case indicators (same vs. different), AAF vs. B5
  and AAF vs. B6, Holm-adjusted over the two comparisons, all 96 cases.
- **H10.2: run-to-run stability under Qwen.** Proportion of cases with an identical action in every run, per arm.
- **H10.3: accuracy under Qwen.** Correct actions (majority of runs) for AAF vs. B5, AAF vs. B6, AAF vs. B1; exact McNemar,
  reported per key; no correction claimed beyond Holm within each key.
- **H10.4: containment.** Count of case-runs with malformed or schema-invalid LLM output per arm, and for AAF the number of
  those in which the final action differs from B1's deterministic action.
- AAF may fail H10.1 (if its actions shift with the model as much as B5/B6) or H10.3; both outcomes will be reported.

## Execution notes added before the main run
- Smoke runs (labels `smoke`, `smoke2`) are pipeline checks only. In `smoke`, Ollama 0.9.0 returned bodies the runner
  could not parse when thinking was on (all B6 calls failed); the runner now tolerates streamed bodies and the main run
  uses Ollama 0.40.9 (the version installed in `smoke2`, where B6 calls succeeded). Model digest
  `500a1f067a9f…b8b41` (qwen3:8b, Q4_K_M).

## Execution
Workflow `.github/workflows/model-replication-qwen.yml` (matrix of shards). A smoke run (`LIMIT` > 0, label `smoke`) checks
the pipeline and timing and is not analysed. The full run uses label `main`. Results are committed by the workflow.
