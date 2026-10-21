# Pilot amendment log

Every change made after the first run of the pilot is recorded here, with the reason.
Changes that tune a reader toward specific cases are not allowed.

| ID | Date | File | Change | Reason |
| --- | --- | --- | --- | --- |
| A1 | 2026-10-21 | regex_reader.py | Removed bare `0` from the negation word list | Bug: version strings such as `3.2.0` were read as negation (seen in TX-E02 during the mock dry run, before any LLM call). This is a defect fix, not a new pattern. |
