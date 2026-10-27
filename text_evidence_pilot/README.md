# Text-evidence development pilot

**Question:** when decision-relevant evidence exists only in messy text (incident notes, logs, scan reports, chat), does an LLM *reader* recover it better than a regex reader — while the frozen deterministic AAF rules keep final authority?

Design idea: **the AI reads, the rules decide.** The LLM never proposes an action. It returns field values with a verbatim quote; a deterministic validator re-checks every value against its quote; only accepted values reach the unchanged AAF rules.

> Development pilot, 21 cases. Not confirmatory, not pre-registered, never pooled with the frozen 32/16-case studies. Its purpose is to decide whether a confirmatory study is worth running and how to design it.

## Arms

| Arm | What reads the text | What decides |
| --- | --- | --- |
| T0 text-blind | nothing (text ignored) | frozen AAF rules |
| T1 regex | `regex_reader.py` (keywords, synonyms, units, negation window, last mention wins) | frozen AAF rules |
| T2 LLM | `llm_reader.py` (gpt-5.6-luna, reasoning none by default), 5 runs per case | frozen AAF rules |
| Ceiling | ground-truth values | frozen AAF rules |

Both readers pass through the same `validator.py` gate and may only fill fields that are **missing** in the structured telemetry. Decision accuracy is agreement with the ceiling decision, i.e. "did the reader recover what the rules needed". The pilot oracle is therefore rules-on-true-evidence; a confirmatory study needs the external policy oracle and independent audit from the revision plan.

## Cases (`cases_text_pilot_v0.yaml`)

| Tier | n | What makes it hard |
| --- | --- | --- |
| easy | 6 | explicit values in standard wording |
| medium | 6 | seconds vs ms, number words, values split across documents, GitOps wording for config drift, "CPU sitting at" |
| hard | 7 | peak vs current value, out-of-scope service mentioned last, negation with an older fixed count, counting restarts from raw log lines, later document supersedes earlier, several fields in one narrative |
| control | 2 | healthy text; irrelevant standup notes (nothing should be extracted) |

## Run

```bash
# offline dry run (no API key; mock reader returns ground truth plus one fabricated value per case)
python -m pytest -q tests/test_text_evidence_pilot.py
python -m text_evidence_pilot.run_pilot --mock --repeats 1

# real pilot
export OPENAI_API_KEY=...            # or use the GitHub Actions workflow below
python -m text_evidence_pilot.run_pilot --repeats 5
```

GitHub Actions: the workflow runs automatically on pushes to the `text-evidence-pilot` branch (or via **Actions → Text-evidence development pilot → Run workflow**). It uses the existing `OPENAI_API_KEY` secret, runs the integrity tests first, commits `text_evidence_pilot/results/` back to the branch and also uploads it as an artifact.

Outputs per run: `summary.json`, `case_results.csv`, `raw_runs.jsonl` (every model output, validation result and decision), `metadata.json` (model, prompt hash, cases hash, run id).

## Known results without any LLM (regex vs text-blind)

- Regex fields: 14 of 25 correct, 4 wrong, 7 missed.
- Decisions matching the ceiling: text-blind 15/21, regex 16/21.
- In TX-H03 and TX-H05 the regex **made the decision worse** than ignoring the text: a real quote, misread (negation; out-of-scope service). The quote check cannot catch a semantic misreading. Watch for the same failure in the LLM arm.

## How to read the pilot result

- **Field level is the main signal.** Text changes the decision in only 6 of 21 cases, so decision counts are small. A confirmatory benchmark should be built so the text-only evidence matters for the decision.
- **Go signal:** the LLM beats regex clearly on medium and hard fields, with few wrong values and no harmful flips on the hard cases, and the validator rejects fabrications.
- **No-go signal:** the LLM ties regex, or misreads hard cases as often as regex does. Then the paper keeps option 1 (authority separation, containment, honest B2 result) without a text-reading claim.
- Report rejected extractions by reason, wrong-but-accepted values, run-to-run variation and token cost alongside accuracy.

## Limitations

- Cases and regex baseline were written by the same person in one session (regex first, then cases). See `AMENDMENTS.md` for the one post-run fix.
- 21 synthetic cases; English only; one model.
- Ground truth is the case author's reading of the text; it has not been independently checked.
