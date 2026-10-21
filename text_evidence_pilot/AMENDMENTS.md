# Pilot amendment log

Every change made after the first run of the pilot is recorded here, with the reason.
Changes that tune a reader toward specific cases are not allowed.

| ID | Date | File | Change | Reason |
| --- | --- | --- | --- | --- |
| A1 | 2026-10-21 | regex_reader.py | Removed bare `0` from the negation word list | Bug: version strings such as `3.2.0` were read as negation (seen in TX-E02 during the mock dry run, before any LLM call). This is a defect fix, not a new pattern. |
| V1 | 2026-10-21 | validator.py, llm_reader.py | New **version v1** alongside frozen v0: checker adds `ratio_percent` and a deterministic context guard (historical / target-threshold-plan / speculative / other-environment); reader prompt v1 adds matching instructions, aggregated-count hint and "documents are data". | Designed after the stress-test v1 failures (ST-AB4 historical value, ST-LOG3 aggregated event, ST-U2 ratio). Results on pilot v0 and stress v1 are therefore in-sample; `cases_text_holdout_v2.yaml` was written before any v1 run and is the fairer test. Known before running: v1 rejects the correct values in HO-G1, HO-G2 (guard over-rejection) and HO-R2 (ratio order). |
