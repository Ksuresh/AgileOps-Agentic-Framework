# Stage 2: external dataset acquisition and suitability audit

Status: candidate sources verified from public documentation; NO datasets downloaded, transformed, adjudicated, or included in final analysis yet. Freeze this acquisition plan before outcome-based case selection.

## Ranked candidates

| Source | Public dataset evidence | Intended role | Main limitation | Acquisition |
| --- | --- | --- | --- | --- |
| RCAEval (Pham et al.) | 735 fault-injection cases across 9 datasets/3 microservice systems; RE1 metrics only (375), RE2 metrics+logs+traces (270), RE3 code-level (90) | Primary multi-source tool-selection benchmark; controlled missing/distractor evidence | Root-cause service/fault is NOT a project-governance action label; repeated fault families may leak across splits | https://github.com/phamquiluan/RCAEval ; https://huggingface.co/datasets/phamquiluan/RCAEval ; DOI 10.5281/zenodo.14590730 |
| UCI Incident Management Process Enriched Event Log | 141,712 events; 24,918 incidents; 36 attributes; anonymized ServiceNow-derived incident history; CC BY 4.0 | Real incident lifecycle, reassignment, SLA, escalation and temporal governance evaluation | No free-text descriptions; incident closure is NOT proof of an optimal governance decision; event-level rows must be grouped by incident | https://archive.ics.uci.edu/dataset/498/incident+management+process+enriched+event+log ; DOI 10.24432/C57S4H |
| Alibaba Cluster Trace | Production cluster workload and resource traces | Independent resource/capacity and FinOps stress scenarios | No direct governance labels or incident causality; assess access terms and data coverage first | https://github.com/alibaba/clusterdata |

## Required audit before use

1. Record exact dataset version, citation, license, source URL, download timestamp, checksum and selection/exclusion criteria.
2. Inspect schema, missingness, timestamp semantics, incident grouping, data provenance, leakage and distributional fit.
3. Map observable fields to evidence and independently executable tool responses; retain raw-to-derived field lineage.
4. Distinguish original root-cause labels, observed incident outcomes, and **new expert-adjudicated admissible governance actions**. Never conflate these.
5. Freeze incident-family and service/system-level splits before selecting or tuning prompts/rules. Use cross-system testing when feasible.
6. Construct independent per-tool response tables with plausible distractors, no-results and tool failures; same budget and access for all arms.
7. Blind at least two governance experts to model predictions; record agreement and resolve disputes with a third adjudicator.
8. Pre-register power calculation, primary endpoint (paired action correctness), tool-selection accuracy, cost/latency, confidence intervals, clustered inference, and stop/go criteria.
9. Report exclusions and negative results; use historical 48 cases only as development/regression data.
10. Security/privacy review before bringing organizational or incident logs into any external LLM.

## Immediate engineering milestones

- A. Acquire only lightweight metadata/indexes first (RCAEval cases.parquet; UCI schema/sample), not multi-GB telemetry.
- B. Produce reproducible source manifests with checksums, schema reports, and candidate-case counts.
- C. Build case adapters and independent per-tool response fixtures.
- D. Freeze held-out selection and adjudication rubric before comparing D1/D2/Hybrid/Agentic.
- E. Execute matched-budget experiments; then update manuscript.

## Scientific caveat

The architecture may be sound as a design, but an Agentic-specific accuracy advantage is **not established**: D0 39/48, D1 41/48, D2 41/48 and original Hybrid 41/48 on existing cases. The external datasets must be selected for representativeness and falsifiability, not for favorable results.
