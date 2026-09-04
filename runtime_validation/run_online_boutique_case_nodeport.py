from __future__ import annotations

"""NodePort provenance wrapper for the frozen Online Boutique scenario runner.

This wrapper changes only the evidence-source description emitted by the
benchmark-specific HTTP measurement adapter. It does not alter scenario
interventions, AAF policy, thresholds, utility logic, or action evaluation.
"""

from runtime_validation import run_online_boutique_case as impl

_original_measure_http = impl.measure_http


def _measure_http_nodeport(url: str, duration_s: int, concurrency: int) -> dict:
    result = _original_measure_http(url, duration_s, concurrency)
    result["source"] = (
        "direct HTTP measurements through the stable Kubernetes NodePort "
        "exposing the Online Boutique frontend Service"
    )
    return result


impl.measure_http = _measure_http_nodeport


if __name__ == "__main__":
    impl.main()
