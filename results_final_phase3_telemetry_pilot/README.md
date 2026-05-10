# Phase 3 Sock Shop Telemetry Pilot

This folder contains the frozen Phase 3 benchmark telemetry pilot outputs for the AgileOps Agentic Framework.

## Evaluation Scope

Phase 3 evaluates AAF on a minimal benchmark telemetry pilot using local runtime artifacts collected from the Sock Shop microservices benchmark.

The pilot uses raw Docker Compose runtime artifacts, including:

- container status
- compose service status
- Docker resource snapshots
- service logs

## Cases

- P3-T01: Sock Shop runtime startup governance check
- P3-T02: Sock Shop catalogue service degradation
- P3-T03: Sock Shop front-end resource scaling / cost proxy

## Key Results

- Benchmark: Sock Shop
- Cases evaluated: 3
- Domain match rate: 1.00
- Action match rate: 1.00
- Evaluation type: local runtime artifact telemetry pilot

## Interpretation

This pilot does not claim production validation. It demonstrates that AAF can consume telemetry derived from real benchmark runtime artifacts and produce domain/action decisions beyond purely generated scenarios.

## Reproducibility

Raw artifacts are stored under:

```text
telemetry_pilot/raw/
