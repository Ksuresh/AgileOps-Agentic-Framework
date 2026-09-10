# Bounded LLM Faithfulness Evaluation

This study evaluates only the optional PM natural-language interface. The authoritative AAF governance action is computed before the LLM call and is never delegated to the model.

## Frozen execution

- GitHub Actions run: `34325587289`
- Head SHA: `dce8bcc8560c0390af4524c8dd0ea286f74d9a5a`
- Artifact: `llm-faithfulness-gpt-5-6-luna`
- Artifact ID: `10093805818`
- Artifact SHA-256: `f0ccfd670129b1598b67b6e455a997453ac0648efd7b4204e9265a60f25f76d7`
- Provider/model: OpenAI `gpt-5.6-luna`
- API: Responses API
- Reasoning effort: `none`
- Tools: disabled
- OpenAI Python SDK: `3.10.0`
- Prompt SHA-256: `3089112b88db9112cf111349f1d9f0e5000c110ee2d0a390bc4e1cbc50fc8f19`
- Test size: 5 frozen AAF decision records x 20 prompts = 100 prompts

Prompt categories: 30 normal, 20 ambiguous, 25 adversarial authority-override attempts, and 25 unsupported-evidence/fabrication prompts.

## Frozen primary metrics

The original workflow artifact contains the raw responses and the original lexical scorer outputs. The original lexical scorer intentionally erred on the conservative side and therefore counted some explicit rejections as apparent violations/repetitions.

A deterministic post-hoc rescoring is provided in `llm_evaluation/rescore_frozen_faithfulness.py`. It operates on the same frozen `faithfulness_outputs.jsonl`; it performs no LLM calls and changes no prompts or responses.

Verified rescored metrics:

| Metric | Result |
|---|---:|
| Exact intent mapping | 91/100 (91%) |
| Raw authoritative-action preservation | 100/100 (100%) |
| System-level authoritative-action preservation | 100/100 (100%) |
| Adversarial authority resistance | 25/25 (100%) |
| Narrative authority violations after contextual rescoring | 0/25 (0%) |
| Unsupported-evidence lure rejection | 25/25 (100%) |
| Unsupported-evidence lure acceptance | 0/25 (0%) |
| Unsupported numeric claims after normalization/context handling | 0/100 (0%) |

The unsupported-evidence cases comprise 19 responses that repeated a lure only to reject it and 6 that rejected the unsupported premise without repeating it.

## Reproduce the rescoring

Download/extract the frozen workflow artifact so that `faithfulness_outputs.jsonl` is in an input directory, then run:

```bash
python llm_evaluation/rescore_frozen_faithfulness.py <frozen-artifact-dir> paper_results/llm_faithfulness/generated_rescore
```

The script produces `faithfulness_rescored_summary.json`, `faithfulness_rescored_outputs.jsonl`, and `faithfulness_rescored_case_results.csv`.

## Interpretation boundary

These are controlled bounded-interface tests, not production-user conversations. The result supports decision preservation and evidence faithfulness for the tested prompts/model/configuration; it must not be described as proving that hallucination is impossible or eliminated.
