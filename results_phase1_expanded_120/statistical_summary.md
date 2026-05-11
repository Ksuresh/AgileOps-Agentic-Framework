# Statistical Summary

## Method Metrics with 95% Confidence Intervals

| Method | Domain Accuracy | 95% CI | Action Match | 95% CI | Action N |
|---|---:|---:|---:|---:|---:|
| Traditional Baseline | 0.633 | [0.547, 0.720] | 0.000 | [0.000, 0.000] | 0 |
| Single-Agent Baseline | 0.692 | [0.609, 0.774] | 0.000 | [0.000, 0.000] | 0 |
| AAF Full | 0.875 | [0.816, 0.934] | 0.792 | [0.719, 0.864] | 120 |
| AAF w/o Consensus | 0.808 | [0.738, 0.879] | 0.792 | [0.719, 0.864] | 120 |
| AAF w/o RAR | 0.808 | [0.738, 0.879] | 0.792 | [0.719, 0.864] | 120 |
| AAF w/o Utility | 0.875 | [0.816, 0.934] | 0.000 | [0.000, 0.000] | 120 |

## Paired Difference Estimates

| Comparison | Mean Difference | 95% CI |
|---|---:|---:|
| AAF Full domain accuracy minus Traditional Baseline | 0.242 | [0.138, 0.345] |
| AAF Full domain accuracy minus Single-Agent Baseline | 0.183 | [0.080, 0.287] |
| AAF Full domain accuracy minus AAF w/o Consensus | 0.067 | [0.016, 0.117] |
| AAF Full domain accuracy minus AAF w/o RAR | 0.067 | [0.016, 0.117] |
| AAF Full action match minus AAF w/o Utility | 0.792 | [0.719, 0.865] |

Note: Confidence intervals are descriptive uncertainty estimates for the controlled evaluation set and should not be interpreted as claims of production-scale generalization.
