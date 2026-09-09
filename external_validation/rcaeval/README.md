# RCAEval External Validation

This directory contains the independent external-fault validation layer for the AAF manuscript, using recorded microservice fault evidence from the public RCAEval benchmark.

## Scope

The selected source is **RCAEval RE2-TrainTicket (RE2-TT)**: 90 cases spanning CPU, memory, disk, delay, packet-loss and socket faults with fault-injection timestamps, annotated root-cause services and multi-source telemetry.

AAF does **not** receive the RCAEval fault label or root-cause label as decision input. Those fields are appended only after the decision for experiment stratification and post-hoc audit. RCAEval therefore supplies external operational evidence, not governance-action ground truth.

## Frozen experimental sequence

1. `probe_schema.py` — inspect representative source telemetry and field semantics.
2. `probe_multisource.py` and `probe_trace_timeseries.py` — inspect additional evidence sources used during adapter development.
3. `adapter.py` — frozen, label-blind RCAEval-to-AAF evidence mapping.
4. `FROZEN_PROTOCOL.md` — frozen split, mapping constraints and interpretation boundaries.
5. `run_pilot.py` — 12 adapter-development/pilot cases (two per fault type).
6. `run_heldout.py` — final 78-case held-out evaluation; these cases are not used for adapter development.

Run the held-out study from the repository root with:

```bash
python external_validation/rcaeval/run_heldout.py
```

Generated outputs are written under:

`paper_results/external_rcaeval/generated/heldout/`

The output includes per-case JSON records, `heldout_case_results.csv` and `heldout_summary.json`. Generated outputs are ignored by git and preserved as GitHub Actions artifacts for frozen executions. No downloaded RCAEval telemetry is committed to this repository.

## Methodological constraints

- Fault and root-cause labels are excluded from the AAF decision path.
- The external dataset provides fault evidence; it does not provide AAF governance actions.
- Only verified source fields are mapped into AAF evidence.
- Proxy evidence remains explicitly identified as proxy evidence.
- The adapter does not manufacture deployment, FinOps or security evidence when the source does not provide it.
- Results are reported as governance behavior/materiality response, not RCAEval "action accuracy."

## Frozen held-out provenance

Successful held-out execution: GitHub Actions run `34316286523`, head SHA `88e7c6e87f11cb5f5840f770e22d20d6fbd2ab83`, artifact ID `10090236316`, artifact SHA-256 `1d2213cb7cec656cb168fbec1821969c0db91d53db86b85581fa3aeec420b730`.

The frozen held-out result contains 78 cases: 26 `Mitigate and monitor` and 52 `No action (observe)`. This distribution reflects AAF governance materiality under the mapped evidence and must not be interpreted as fault-detection accuracy.

## Data source

RCAEval: `phamquiluan/RCAEval`, using the public Hugging Face dataset copy for case-wise Parquet downloads. The runner downloads only the files required for execution.
