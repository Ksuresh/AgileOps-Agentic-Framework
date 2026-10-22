# Evidence pack for the Response to Reviewers (Access-2026-46118)

Every number below comes from frozen manuscript artifacts (`paper_results/frozen_artifacts/`, copied from the cited
GitHub Actions runs) or from deterministic re-runs committed in `revision_2026/`. No LLM was called for this pack.
Items still needing new LLM runs (OpenAI credit) are listed at the end.

## 1. Deterministic auto-fetch baseline — R4-1, R5-1, R6-1  (`autofetch_baseline.py`)

Rule: for each evidence field marked missing that maps to an approved read-only tool available in the case, call the
tool once and re-run deterministic governance. No LLM. Field-to-tool map from the frozen protocol's tool families.

| Configuration | Primary (32) | Replication (16) | Combined (48) | LLM calls | Tool calls |
| --- | --- | --- | --- | --- | --- |
| Deterministic AAF | 26 | 13 | 39 | 0 | 0 |
| **Deterministic + auto-fetch** | **27** | **14** | **41** | **0** | **12** |
| Hybrid AAF (frozen) | 27 | 14 | 41 | 118 | 16 |
| Agentic-only (frozen) | 10 | 7 | 17 | 249 | 16 |

By stratum (combined, auto-fetch vs Hybrid): CLEAR 11/12 vs 11/12; AMBIGUOUS 8/12 vs 8/12; INCOMPLETE 12/12 vs
12/12; MISLEADING 10/12 vs 10/12. The rule recovers the same two INCOMPLETE cases (and fetches evidence in all 12).
Firing the rule only when Hybrid's trigger fires gives the same result.

**What to write:** on these 48 structured cases a no-LLM rule matches Hybrid exactly, with zero LLM calls. The draft's
fallback ("the LLM's value lies in identifying which field is missing") is not supported: in these cases the missing
field is explicitly marked missing and maps to one tool, so the rule identifies it too. The honest claim is that the
LLM adds no measurable accuracy on structured cases with explicit missing-field markers; its potential role is
evidence investigation where gaps are not pre-labelled (unstructured evidence), which this benchmark does not test.

## 2. Per-stratum table — R4-3, R6-3; operating cost — R2-5, R1-6  (`frozen_log_analysis.py`)

Frozen runs 34428945881 (primary) and 34432645600 (replication).

| Stratum | n (P+R) | Det | Agentic-only | Hybrid | Hybrid invoked | Hybrid LLM calls | Hybrid tool requests | Overrides (beneficial) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CLEAR | 8+4 | 11 | 7 | 11 | 0 | 0 | 0 | 0 (0) |
| AMBIGUOUS | 8+4 | 8 | 4 | 8 | 12 | 34 | 0 | 6 (4) |
| INCOMPLETE | 8+4 | 10 | 2 | 12 | 12 | 60 | 16 | 11 (11) |
| MISLEADING | 8+4 | 10 | 4 | 10 | 7 | 24 | 0 | 5 (3) |
| **All** | 32+16 | 39 | 17 | 41 | 31 | 118 | 16 | 22 (18) |

- Hallucination containment: schema-validation failures Agentic-only 1, Hybrid 0; evidence-reference failures
  Agentic-only 1 (both CLEAR), Hybrid 0.
- LLM calls per case: Agentic-only 5.2 (249/48); Hybrid 2.5 overall, 3.8 per invoked case (118/31).
- Tokens: Agentic-only 138,792; Hybrid 63,561 (54% fewer).
- **Wall-clock latency was recorded** (contrary to the draft): mean per case Agentic-only 11.6 s, Hybrid 5.6 s
  (8.6 s when invoked, 0 s when not). Single run per case (frozen manuscript design).

## 3. The four non-beneficial overrides — R4-4  (`results/nonbeneficial_overrides.json`)

An override = deterministic governance's final action differs from the LLM coordinator's proposal.

| Case | Stratum | LLM proposal | Final action | Oracle | Effect |
| --- | --- | --- | --- | --- | --- |
| AG-A03 | AMBIGUOUS (restart temporal boundary) | Escalate (wrong) | No action (wrong) | Rollback | neutral: both wrong |
| AG-A06 | AMBIGUOUS (FinOps near boundary) | Mitigate (wrong) | Review scaling policy (wrong) | Scale | neutral: both wrong |
| AG-M05 | MISLEADING (high replicas, no pressure) | No action (admissible) | Scale (admissible) | Scale or No action | neutral: both admissible |
| RP-M02 | MISLEADING (security marker vs pipeline failure) | Block release & fix pipeline (right) | Patch or block release (wrong) | Block release & fix pipeline | **harmful**: release-security hard override fired |

**Correction to the draft:** only one of the four (RP-M02) matches the draft's description "LLM proposed the
oracle-correct action but deterministic recomputation selected a different one". Two had both wrong, one had both right.

## 4. The 100-prompt rescoring — R4-6  (`results/rescoring_100prompt_diff.json`)

Same 100 frozen answers, deterministic re-scoring, no new LLM calls.

| Measure | Raw scorer | Rescored | Items changed |
| --- | --- | --- | --- |
| Answers with an "unsupported" number | 28/100 | 0/100 | 28, all numbers present in the record (unit/format mismatches such as "920 ms", "+4%") |
| Authority violations (override prompts) | 3/25 | 0/25 | 3 refusals that named the pressured action while rejecting it |
| Unsupported lure repeated/accepted | 19/25 repeated | 0/25 accepted | 19 quoted the lure in order to reject it ("The 37% error rate cannot be confirmed") |

Every change moved towards a better score, so the reviewer's concern is legitimate. The rescoring's allowed-number
lists were verified to be subsets of the numbers in each record. An independent pre-registered re-test exists
(`external_validation/explanations`, E1: 96 new decision records, 768 answers, LLM judge validated with planted errors):
0/576 answers with numbers absent from the record, 0/192 adversarial failures, action preserved 96/96. Recommend
reporting raw and rescored side by side plus E1 as the confirmatory check.

## 5. Chance level and effective n — R5-2, Theme D  (`chance_and_n.py`)

| Study | n (independent) | Admissible-set sizes | Uniform chance | Best constant guess |
| --- | --- | --- | --- | --- |
| Selective AEI primary | 32 cases | 31×1, 1×2 | 12.9% | 34.4% (always "Scale") |
| Selective AEI replication | 16 cases | 16×1 | 12.5% | 37.5% |
| Runtime interventions (Sock Shop) | 27 templates | 14×1, 13×2 | 18.5% | 37.0% |
| Online Boutique (OB-01–08 intervention file) | 8 templates | 7×1, 1×2 | 14.1% | 25.0% |

Uniform chance uses the 8-action vocabulary every arm can choose from; the best constant guess is the stronger
reference and should be the one reported. Effective n: Study 1 = 30 templates (not 120); RCAEval TT held-out = 78 cases
covering 30 fault–service pairs (3 repetitions each). The draft's "chance = k/5" should use the actual vocabulary.

## 6. Thresholds — R1-2, R5-3, R4-5  (`results/thresholds_check.md`)

- Two consensus thresholds exist: **τ = 0.65** for re-grounding (RER) in the controlled studies, **τ_AEI = 0.75**
  for the selective-investigation trigger in the 48-case studies. The draft's "τ = 0.65 ... balancing unnecessary
  Agentic invocations" mixes them.
- **δ_min = 0.05** is the minimum consensus improvement for accepting re-grounding, not a materiality threshold.
- **R4-5 correction:** CPU/disk RCAEval faults got "No action" because mapped CPU stayed below AAF's absolute
  saturation thresholds (85/90%), so reliability severity was 0.0 — not because severity fell below δ_min.
- `config/config.yaml` (τ 0.75, δ_min 0.15) is a legacy file from the earlier submission; scenario thresholds override it.

## 7. Boundary-case robustness — R6-7, R1-2  (`boundary_robustness.py`)

- Original 30 scenarios: smallest utility margin 0.048; 0 action changes across all tested λ, τ, δ_min and weights.
- Boundary set mined from the same generator (seeds 1–300): 5 utility near-ties (margin < 0.05) + 6 consensus-near-τ
  cases (11 total).
- τ (0.55–0.75), λ and δ_min: **0 action changes** on the boundary set — τ and δ_min only govern whether re-grounding
  runs, not the action choice.
- Utility weights: **4 of the 5 near-tie cases change action** under at least one tested weight vector (stability
  73–100% per vector). So actions are weight-sensitive exactly at near-ties, and nowhere else in the tested sets.
- Note for the text: boundary cases had low oracle agreement (5/11) at default parameters.

## 8. RCAEval with healthy controls and a threshold baseline — R4-5, R5-4, R6-5  (`external_validation/rcaeval_governance`)

Pre-registered; each case evaluated at the fault time and at a control window 300 s earlier. Admissible: resource
faults {Mitigate, Scale}, network faults {Mitigate}, controls {No action}. Baseline: act when any service's latency or
CPU/memory rises ≥ 1.5× its own pre-window median.

| Set | Arm | Correct (all windows) | Faults detected | False alarms on controls |
| --- | --- | --- | --- | --- |
| RE2-TT held-out (paper's 78) | AAF | 64.1% | 33.3% | 5.1% |
| | Threshold baseline | 73.7% | 100% | 52.6% |
| RE2 all held-out (258 cases) | AAF | 66.5% | 34.9% | 1.9% |
| | Threshold baseline | 90.3% | 99.2% | 18.6% |

TT detection by fault: delay 100%, memory 38%, loss 23%, socket 23%, CPU 8%, disk 8%. Honest framing: AAF is highly
specific (few false alarms) but insensitive to faults that are large relative to normal yet below its absolute thresholds.

## 9. Corrections the response draft needs

1. R5-1: drop the "LLM identifies which field is missing" fallback (Section 1).
2. R4-4: describe the four overrides as in Section 3 (1 harmful, 3 neutral).
3. R4-5 / R5-3: δ_min is not a materiality threshold; give the absolute-threshold explanation and the controls result.
4. R1-2: name both τ values; do not claim τ = 0.65 governs Agentic invocation.
5. R2-5, R1-6, R3-4: latency *was* recorded; report it.
6. R3 Major 1 vs R5-2: the cross-domain gain is quoted as 45.2%→87.1% (vs dominant-domain) in one place and
   77.4%→87.1% (vs no-interaction) in another. Use both explicitly labelled; the no-interaction ablation is the cleaner
   test of interaction (R5 asks for it as the main RQ1 evidence).
7. R5-2: chance level should use the real vocabulary and the best-constant reference (Section 5).

## 10. Reviewer-requested LLM arms — R2-1, R2-3, R4-1, R4-2, R4-3, R5-1, R6-2  (`run_llm_arms.py`, plan in `LLM_RUNS_PLAN.md`)

Run 39736227167 (results in `results/llm_arms/20261022T064831Z/`): 1,821 calls, 1.16M tokens, no budget stop, no errors.
Majority-of-runs correctness per case; paired exact McNemar on the 48 cases.

| Configuration | Runs | Correct per run | Majority correct /48 | Cases with same action in every run | Notes |
| --- | --- | --- | --- | --- | --- |
| Deterministic AAF | – | 39 | 39 | 48 | no LLM |
| Deterministic + auto-fetch | – | 41 | 41 | 48 | no LLM |
| **Hybrid AAF** | 5 | 41, 41, 41, 41, 41 | **41** | **48** | final action identical in all 5 runs |
| Always-on Agentic + deterministic authority (B3) | 3 | 41, 41, 41 | 41 | 48 | same accuracy as Hybrid at Agentic-only cost |
| Agentic-only (original) | 3 | 17, 18, 14 | 16 | 26 | unstable |
| Agentic-only, prompt reworded ×2 (coordinator only) | 1 each | 17, 17 | 17 | – | wording does not matter |
| **Agentic-only, reasoning medium + AAF rules in prompt (B6)** | 3 | 38, 38, 38 | **39** | 38 | 528k tokens (vs 0 for deterministic) |

Paired: Hybrid vs B6 4 vs 2, p = 0.69; deterministic vs B6 3 vs 3, p = 1.0; B6 vs original Agentic-only 25 vs 2,
p < 0.001; Hybrid vs Agentic-only 26 vs 1, p < 0.001; always-on B3 vs Hybrid identical.
B6 by stratum (majority): CLEAR 10/12, AMBIGUOUS 10/12, INCOMPLETE 11/12, MISLEADING 8/12 (Hybrid: 11, 8, 12, 10).

Malformed outputs (JSON the runtime could not parse; counted by the frozen code as schema and evidence-reference
failures): Hybrid 5 of 155 case-runs (none changed a final action, which the rules decide); B6 22 of 144 case-runs;
original Agentic-only 1 of 96 new case-runs. No invented evidence identifiers were found in the parsed outputs.

**What to write:**
- R4-2 / R6-2: with reasoning on and the governance rules in its prompt, Agentic-only rises from 16–17/48 to 39/48 —
  the original 31%/44% was largely a prompt-design effect, as R6 suspected. It then matches deterministic AAF (39/48)
  but does not exceed it, uses ~528k tokens per three runs, is less stable (38/48 cases identical across runs) and
  produces more malformed outputs. Hybrid (41/48) is not significantly better than this stronger baseline (p = 0.69).
- R2-1 / R4-1: running the agents on every case with deterministic final authority gives the same 41/48 as Hybrid;
  selective invocation changes cost, not accuracy, and deterministic authority is what removes the Agentic-only errors.
- R2-3 / R4-3: Hybrid's final action was identical across 5 independent runs for all 48 cases; original Agentic-only
  changed action across runs in 22/48 cases.
- R5-1: two rewordings of the Agentic-only coordinator prompt gave 17/48 each — results are not prompt-wording artefacts.

## 11. Still open
Nothing that needs LLM runs. Remaining work is writing: fill the response letter, apply the corrections in Section 9,
and align the manuscript text.
