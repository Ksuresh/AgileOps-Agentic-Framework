# Reproducibility Summary
Scenarios: 30
## AAF Latency Breakdown (No RAR Triggered)
| Stage | Mean (ms) | Std | P50 | P95 | P99 |
|---|---:|---:|---:|---:|---:|
| T-IN | 45.9 | 6.5 | 46.6 | 54.6 | 55.8 |
| AG-INF | 93.9 | 11.3 | 94.7 | 111.1 | 119.9 |
| CN-CHK | 31.1 | 3.8 | 31.2 | 35.9 | 36.3 |
| LLM-XP | 104.9 | 10.9 | 104.7 | 122.5 | 127.9 |
| OUT-GEN | 23.6 | 3.6 | 23.6 | 29.2 | 32.1 |
| TOTAL (No RAR) | 299.5 | 18.2 | 296.5 | 328.6 | 336.9 |

RAR triggered in 9/30 scenarios (30.0%).

## Simulated Baselines (for paper-comparable comparisons)
| Method | Accuracy | Decision latency (min, mean±sd) |
|---|---:|---:|
| Traditional | 0.57 | 11.4 ± 1.7 |
| Single-agent LLM | 0.67 | 7.0 ± 1.5 |
| AAF | 0.97 | 3.9 ± 1.2 |
