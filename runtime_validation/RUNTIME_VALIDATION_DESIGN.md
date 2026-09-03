# Runtime Validation Design

## Goal

Provide reproducible external-validity evidence beyond the synthetic oracle benchmark by running AAF against **real runtime artifacts from a benchmark microservice system under controlled injected conditions**.

The primary system is Sock Shop. Google Online Boutique is an optional second-system replication after the Sock Shop study is complete.

## Validation logic

Each runtime case follows this sequence:

1. Start from a known healthy benchmark deployment.
2. Apply exactly one controlled intervention or one explicitly defined compound intervention.
3. Record the intervention before AAF execution.
4. Collect raw runtime artifacts.
5. Convert those raw artifacts into structured AAF evidence using a documented mapping layer.
6. Execute AAF without access to the intervention label.
7. Compare the predicted domain/action with the pre-specified runtime oracle derived from the intervention.
8. Restore the benchmark to the healthy baseline before the next case.

This separation is essential: **intervention -> oracle** is fixed independently of **runtime artifacts -> AAF evidence -> AAF decision**.

## Primary Sock Shop case matrix

| ID | Controlled intervention | Oracle domain(s) | Admissible governance action(s) | Primary purpose |
|---|---|---|---|---|
| RT-01 | Healthy baseline | None / Observe | No action (observe) | False-positive control |
| RT-02 | Stop catalogue service | SRE | Mitigate and monitor | Runtime availability degradation |
| RT-03 | Stop carts service | SRE | Mitigate and monitor | Independent service degradation |
| RT-04 | Repeatedly restart front-end | DevOps, SRE | Rollback to stable deployment; Mitigate and monitor | Post-change instability |
| RT-05 | Deploy invalid/mismatched image tag | DevOps | Block release and fix pipeline; Rollback to stable deployment | Deployment artifact failure |
| RT-06 | Introduce invalid configuration/environment value | DevOps | Rollback to stable deployment; Block release and fix pipeline | Configuration drift/failure |
| RT-07 | Constrain CPU for front-end | SRE | Mitigate and monitor; Scale adjustment | Saturation/performance degradation |
| RT-08 | Constrain memory for front-end | SRE | Mitigate and monitor; Scale adjustment | Resource-pressure degradation |
| RT-09 | Scale front-end replicas substantially above baseline | FinOps | Scale adjustment; Review scaling policy | Resource-footprint/cost proxy |
| RT-10 | Scale service up while load remains baseline | FinOps | Scale adjustment; Review scaling policy | Over-provisioning control |
| RT-11 | Inject high request load | SRE | Mitigate and monitor; Scale adjustment | Load-driven reliability pressure |
| RT-12 | High load + excessive scale-out | SRE, FinOps | Scale adjustment; Mitigate and monitor | Cost/reliability trade-off |
| RT-13 | Failed deployment + elevated error/latency | DevOps, SRE | Rollback to stable deployment | Cross-domain deployment incident |
| RT-14 | Policy/security scan failure on release artifact | DevSecOps | Patch or block release | Security gate failure |
| RT-15 | IAM/policy configuration drift | DevSecOps | Patch or block release | Security-policy drift |
| RT-16 | Security issue + otherwise healthy runtime | DevSecOps | Patch or block release | Security-only control |
| RT-17 | Security issue + high latency | DevSecOps, SRE | Patch or block release; Mitigate and monitor | Security/reliability conflict |
| RT-18 | Failed deployment + security issue | DevOps, DevSecOps | Patch or block release; Block release and fix pipeline | Release-governance conflict |
| RT-19 | Missing one evidence source after intervention | Depends on intervention | Same as parent intervention | RER recovery test |
| RT-20 | Noisy/partial evidence after intervention | Depends on intervention | Same as parent intervention | Robustness test |

## Raw artifacts to collect per case

Every case directory should preserve raw evidence before transformation:

- `docker_ps.txt`
- `compose_ps.txt`
- `docker_stats.txt`
- relevant service logs
- container inspect output for affected services
- benchmark configuration / compose diff used for the intervention
- intervention metadata JSON
- timestamps for intervention, collection start, collection end, and recovery
- optional load-generator output
- optional security-scanner output

Do not report derived AAF fields as raw measurements.

## Evidence transformation rules

The runtime study must distinguish three evidence levels:

1. **Raw artifact** — directly collected from the running benchmark.
2. **Derived observable** — calculated from raw artifacts, e.g. restart count, unavailable-service indicator, replica count, observed error count.
3. **AAF normalized evidence** — values consumed by the deterministic domain analyzers.

Where exact metrics are unavailable, use transparent categorical/bounded mappings rather than invented continuous values. For example, prefer:

- `service_unavailable=True`
- `restart_count=12`
- `replica_count=8`

rather than manufacturing p95 latency or cloud-cost percentages from unrelated signals.

If the current AAF schema requires a continuous field, the transformation function must document the proxy rule and the manuscript must label it as a proxy.

## Repetition

Run each primary intervention at least **3 independent repetitions** after restoring the healthy baseline. A 20-case matrix with 3 repetitions yields up to 60 runtime executions.

For expensive or manual security cases, a smaller repetition count may be reported explicitly rather than silently mixing sample sizes.

## Metrics

Report:

- domain-oracle agreement
- action-oracle agreement
- Wilson 95% confidence intervals
- paired McNemar comparison against the dominant-domain baseline when both policies operate on the same cases
- RER trigger rate
- RER decision recovery / stability under RT-19 and RT-20
- mean, P50, P95, P99 stage latency
- additional latency per RER loop
- failure-case count and qualitative error categories

Do not describe these as production accuracy.

## Failure analysis

Every mismatch must be assigned to one of these categories where applicable:

- evidence extraction failure
- normalization/proxy error
- domain-rule error
- consensus/RER error
- utility/action-selection error
- genuinely ambiguous oracle

Include at least 3 representative failure cases in the paper if failures occur.

## Second-system replication

After Sock Shop is stable, replicate a reduced subset on Google Online Boutique:

- one deployment/configuration case
- one reliability case
- one resource/cost-proxy case
- one security case
- one cross-domain case

This is intended to test portability of the evidence-to-governance pipeline, not to create another large benchmark.
