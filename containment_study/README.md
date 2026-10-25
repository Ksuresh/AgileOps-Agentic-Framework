# Study 11: LLM-failure containment

**Claim tested.** Deterministic governance keeps LLM failures from changing the governed action: under any LLM failure,
AAF's final action equals the deterministic-only action (B1) unless the LLM retrieved a genuine read-only tool result, in
which case it equals B1 on the evidence augmented with that result. Configurations in which the LLM decides (B5) have no
such bound.

**Design.** The LLM layer is replaced by four fault models (`run_containment.py`), seeded per case/run (CRC32):
`garbage` (unparseable output), `harmful` (always "No action (observe)", confidence 0.99, fabricated evidence ids, no tools),
`inverted` (always "Scale adjustment"), `random` (random action, random tool requests, random confidence; 5 runs).
Cases: Study 6 (48) and Study 7 (48). Because the fault models are deterministic simulations with no tunable parameters,
the study was run once as written; no setting was changed after seeing results.

**Planned extension (real model).** Prompt injection: a free-text evidence field instructing the model to select a fixed
action, run with Qwen3-8B for AAF, B5 and B6 (same workflow as Study 10); outcome = how often the injected action is chosen
as final.

Results: `results/containment_summary.json`, rows in `results/containment_rows.jsonl`.

## Note on protocol order
Unlike the other revision studies, Study 11 has no protocol committed before its run: the fault models, the description and the
results were committed together (commit b2a14ea). The study is a deterministic simulation with no tunable parameters and was run
once as written. Of the 17 random-mode runs in which AAF's action differed from the rules-only action (each caused by a genuine
tool result), 16 turned a wrong action into a correct one and one changed between two correct actions on the authors' keys of
Studies 6 and 7; none made a correct action wrong.
