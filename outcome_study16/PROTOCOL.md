# Study 16: Frozen rule set v1.4 on a second application, Weaveworks Sock Shop (frozen before calibration and before any Study 16 run)

## Purpose
Studies 9-15 measure outcomes on one application, Google Online Boutique. Study 16 repeats the Study 15 design on an
independent microservice benchmark with a different code base, Weaveworks Sock Shop
(github.com/microservices-demo/microservices-demo, commit 9dff06fae4981921caec6a62393a6ebfce4b3e3f, `deploy/kubernetes/complete-demo.yaml`
deployed into the default namespace). It tests whether the Study 14-15 findings depend on the application.

## What is frozen
Everything frozen in Study 15 (rules v1.4, oracle definitions, deciders, prompts, model, runtime, three runs per LLM arm),
including Study 15 Amendment 1 (storefront restart after every backend change). The rules are not edited until Study 16 is
analysed.

## What differs from Study 15, and why
- Application and services: front-end (SF-*, Node.js; the storefront) and catalogue (SC-*, Go; reached through the
  front-end). Measured request: `GET /catalogue` on the front-end (the Sock Shop home page is static HTML and calls no
  backend). Pods are labelled `name=<service>` (`LABEL_KEY=name`).
- Thresholds: the same rule as Study 9, recomputed for this application: reliability violation if p95 > 2 x the median
  healthy p95 (three 45 s healthy measurements, `calibrate16.py`) or errors > 1%; security and cost rules unchanged.
- Crash fault: Sock Shop services take no PORT setting, so the crash release replaces the container entrypoint with one that
  does not exist (`CRASH_MODE=command`); the release signal compares the entrypoint as well.
- Memory: the manifest's memory settings are kept (`KEEP_MEMORY=1`; the Node and Java services need more than the 128Mi
  used for Online Boutique). CPU request (100m) and limit (200m) are as in Studies 9-15, so the cost proxy is unchanged.
- Images: Sock Shop's published tags (docker.io/weaveworksdemos). Selection rules as in Study 15 (vulnerable: >= 1 critical
  CVE more than stable, rolls out, storefront serves, first two; low-risk: no more critical CVEs than stable and more
  HIGH/MEDIUM findings, first one, else stable). If no vulnerable tag qualifies for a service, that service's vulnerable
  cases carry an exposed secret instead (`vuln_fallback`, recorded), so the security domain is still exercised.
- Degradation: CPU limit 50m to 5m, largest that violates; these services have no latency setting, so if none violates 5m
  is used and recorded.

## Cases (`cases.yaml`, 38; 28 fault cases)
The Study 15 per-service design (19 cases: C1, L4, R3, P3, S3, K2, D1, X2) for each of the two services.

## Hypotheses (exact McNemar on paired per-case O1 correctness, fault cases n = 28 unless stated)
- **H16.1 (primary):** AAF-14 vs each siloed expert (D-DevOps-14, D-SRE-14, D-FinOps-14, D-Sec-14, L-DevOps, L-SRE, L-FinOps,
  L-Sec); Holm over 8.
- **H16.2:** AAF-14 vs B5, B6 and AlwaysRollback; Holm over 3.
- **H16.3 (pooled, two applications):** AAF-14 vs B6, B5, AlwaysRollback and D-Sec over the fault cases of Studies 14, 15
  and 16; Holm over 4.
- **H16.4 (risk-based release gate):** on the 8 L cases, share O2-correct with Wilson interval; McNemar vs B5, B6,
  AlwaysRollback, L-Sec (descriptive); pooled with the L cases of Studies 14 and 15.
- **H16.5-H16.7:** security priority, per-service counts and withheld-evidence comparisons, as H15.5-H15.7.
- **Descriptive:** discriminating cases, replicate agreement, LLM usage; calibration results.
Any comparison may go against AAF; every pre-specified comparison is reported. The analysis script
(`analyze16.py`) is committed with this protocol, before any data.

## Execution
Calibration, two measurement replicates (decisions use replicate 1), deciders. Resumed runs use identical settings;
deviations are appended below as execution notes.
