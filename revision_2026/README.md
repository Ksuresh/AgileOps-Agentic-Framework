# Revision analyses for IEEE Access manuscript Access-2026-46118

All analyses requested by the reviewers, with the reviewer comment each one answers. Frozen manuscript artifacts used
as inputs are copied into `paper_results/frozen_artifacts/` from the GitHub Actions runs cited in
`paper_results/MANUSCRIPT_RESULT_MAP.md`. Summary of every number: `EVIDENCE_PACK.md`.

| Script | Reviewer comments | LLM calls | Output (`results/`) |
| --- | --- | --- | --- |
| `autofetch_baseline.py` — deterministic auto-fetch baseline | R4-1, R5-1, R6-1 | 0 | `autofetch_cases.csv`, `autofetch_summary.json` |
| `frozen_log_analysis.py` — per-stratum table, LLM calls, tools, latency, overrides | R1-6, R2-5, R4-3, R4-4, R6-3 | 0 | `frozen_per_case.csv`, `frozen_per_stratum.json`, `nonbeneficial_overrides.json` |
| `llm_evaluation/rescore_frozen_faithfulness.py` (unchanged) + diff | R4-6 | 0 | `rescored_100prompt/`, `rescoring_100prompt_diff.json` |
| `chance_and_n.py` — admissible-set sizes, chance level | R5-2 | 0 | `chance_level.json` |
| `chance_tables.py` — uniform-chance and best-constant rows for manuscript Tables 6–8 and 11 (exact case sets; OB-09–16 admissible sets from frozen commit fdf222fc, copied to `inputs/`) | R5-2 | 0 | `chance_tables.json` |
| thresholds check | R1-2, R4-5, R5-3 | 0 | `thresholds_check.md` |
| `boundary_robustness.py` — boundary-case parameter robustness | R1-2, R6-7 | 0 | `boundary_robustness.json`, `boundary_sweep.csv` |
| `run_llm_arms.py` — repeated Hybrid/Agentic-only runs, always-on + deterministic authority, Agentic-only with reasoning and governance rules, prompt rewordings (plan: `LLM_RUNS_PLAN.md`) | R2-1, R2-3, R4-1, R4-2, R4-3, R5-1, R6-2 | 1,821 | `llm_arms/20261022T064831Z/` (GitHub Actions run 39736227167) |
| `external_validation/rcaeval_governance/` — RCAEval with healthy controls and a threshold baseline (pre-registered) | R4-5, R5-4, R6-5 | 0 | `external_validation/rcaeval_governance/results/` |
| `external_validation/explanations/` — PM-interface faithfulness re-test, E1 (pre-registered) | R2-4, R2-6, R4-6, R6-6 | ~2,300 | `external_validation/explanations/results/20261021T125701Z/` |
| `external_validation/aaf_v3/rcaeval_v3.py` — change-aware materiality adapter (AAF-CA in the manuscript), frozen on RE2, held-out RE1/RE3 (plan: `PLAN.md`, results: `RESULTS.md` H1/H2) | R4-5, R5-4, R6-5 | 0 | `external_validation/aaf_v3/results/` (GitHub Actions run 39600613441) |
| `external_validation/expert_study/` — Study 7: 48 new pre-registered cases (24 standard, 24 unflagged-gap), sealed author key, AI-assisted practitioner labels on 96 cases (amendments A1, A2 in PROTOCOL.md) | R1-3, R1-4, R4-1, R5-1, R6-1, R6-4 | 905 | `external_validation/expert_study/results/20261022T130735Z/`, `analysis_author_key.json`, `analysis.json` (run 39784044792) |
| `external_validation/investigator_fix/` + `agentic_experiments/hybrid_v2.py` — Study 8: pre-registered investigator fix (HY2-F, HY2-L) on 36 fresh generated cases | R4-1, R5-1, R6-1 | 1,488 | `external_validation/investigator_fix/results/20261022T132423Z/`, `analysis_fx.json` (run 39786446655) |
| `PROVENANCE.md` — protocol/run/result timeline for every study | R6-4, R3-4 | 0 | — |

Reproduce (repository root, Python 3.11, `pip install -r requirements.txt`):

```
python revision_2026/autofetch_baseline.py
python revision_2026/frozen_log_analysis.py
python llm_evaluation/rescore_frozen_faithfulness.py paper_results/frozen_artifacts/llm_faithfulness_34325587289 revision_2026/results/rescored_100prompt
python revision_2026/chance_and_n.py
python revision_2026/chance_tables.py
python revision_2026/boundary_robustness.py
OPENAI_API_KEY=... python revision_2026/run_llm_arms.py      # model gpt-5.6-luna; budget caps in the script
```

Other work under `external_validation/` from the same revision period is exploratory and not cited in the revised
manuscript: the public-postmortem study, the text side of AAF v3 (`aaf_v3/text_v3.py`, H3), the explanation prompt
variants (v2, v3) and the text-evidence pilot.
