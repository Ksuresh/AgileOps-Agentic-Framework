# RCAEval External Validation

This directory contains the third validation layer for the AAF manuscript: evaluation on externally generated microservice fault evidence from the public RCAEval benchmark.

## Scope

The selected source is **RCAEval RE2-TrainTicket (RE2-TT)**. RCAEval reports 90 RE2-TT failure cases spanning CPU, memory, disk, delay, packet-loss, and socket faults. Each case has a fault-injection timestamp, an annotated root-cause service, and multi-source telemetry.

AAF does **not** receive the RCAEval fault label or root-cause label as decision input. Those fields are retained only for experiment stratification and post-hoc validation. AAF receives only evidence derived from the recorded telemetry.

## Experimental separation

1. `probe_schema.py` — inspects representative external cases and documents the source telemetry before any AAF adapter is frozen.
2. `adapter.py` — will contain the frozen, provenance-preserving RCAEval-to-AAF evidence mapping after the probe is reviewed.
3. `run_validation.py` — will execute AAF on pre-injection and post-injection evidence windows after the adapter and admissible-action protocol are frozen.
4. `evaluate.py` — will compute the external-validation summary and paired comparisons.

Generated outputs are written only under:

`paper_results/external_rcaeval/generated/`

That path is ignored by git and uploaded by GitHub Actions as the execution artifact. No downloaded RCAEval telemetry is committed to this repository.

## Methodological constraints

- The external dataset provides fault evidence; it does not provide AAF governance actions.
- The RCAEval fault/root-cause labels must never be used by the AAF decision path.
- Only source fields whose semantics and units are verified by the schema probe may be mapped into AAF.
- Proxy evidence must be labelled as proxy evidence; it must not be presented as directly measured cost/security/deployment evidence.
- Admissible governance actions must be frozen independently of AAF outputs before the final validation run.

## Data source

RCAEval: `phamquiluan/RCAEval`, using the public Hugging Face dataset copy for case-wise Parquet downloads. The experiment downloads only the cases required for the run.