from __future__ import annotations

"""Label-blind RCAEval -> AAF evidence adapter.

The adapter scans all service metrics and never receives RCAEval fault labels or
root-cause service annotations. Only source observables with defensible semantics
are mapped into AAF fields. Other signals are retained in diagnostics only.
"""

from dataclasses import dataclass
from typing import Any

import pandas as pd

PRE_SECONDS = 300
POST_SECONDS = 300


@dataclass(frozen=True)
class Change:
    column: str
    pre_median: float
    post_median: float
    ratio: float | None


def _median(series: pd.Series) -> float | None:
    x = pd.to_numeric(series, errors="coerce").dropna()
    return None if x.empty else float(x.median())


def _changes(metrics: pd.DataFrame, inject_time: int, suffix: str) -> list[Change]:
    t = pd.to_numeric(metrics["time"], errors="coerce")
    pre_mask = (t >= inject_time - PRE_SECONDS) & (t < inject_time)
    post_mask = (t >= inject_time) & (t < inject_time + POST_SECONDS)
    out: list[Change] = []
    for col in metrics.columns:
        if col == "time" or not col.lower().endswith(suffix.lower()):
            continue
        pre = _median(metrics.loc[pre_mask, col])
        post = _median(metrics.loc[post_mask, col])
        if pre is None or post is None:
            continue
        ratio = None if pre == 0 else post / pre
        out.append(Change(col, pre, post, ratio))
    return out


def _largest_ratio(changes: list[Change]) -> Change | None:
    usable = [c for c in changes if c.ratio is not None]
    return max(usable, key=lambda c: float(c.ratio), default=None)


def _largest_post(changes: list[Change]) -> Change | None:
    return max(changes, key=lambda c: c.post_median, default=None)


def adapt(metrics: pd.DataFrame, inject_time: int) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return (AAF telemetry, adapter diagnostics) without label/oracle input."""
    if "time" not in metrics.columns:
        raise ValueError("RCAEval metrics require a time column")

    latency90 = _changes(metrics, inject_time, "_latency-90")
    cpu = _changes(metrics, inject_time, "_cpu")
    mem = _changes(metrics, inject_time, "_mem")
    disk = _changes(metrics, inject_time, "_diskio")
    socket = _changes(metrics, inject_time, "_socket")
    error = _changes(metrics, inject_time, "_error")

    lat_ratio = _largest_ratio(latency90)
    lat_abs = _largest_post(latency90)
    cpu_ratio = _largest_ratio(cpu)
    cpu_abs = _largest_post(cpu)

    telemetry: dict[str, Any] = {
        "deploy": {"_evidence": {}},
        "sre": {"_evidence": {}},
        "finops": {"_evidence": {}},
        "sec": {"_evidence": {}},
    }

    # RCAEval exposes service latency-90, not p95. Because p95 >= p90, a measured
    # p90 crossing an AAF p95 threshold is a conservative lower-bound proxy. We
    # only expose it when the post-injection median is at least 1.5x its own
    # pre-injection baseline, preventing unrelated high-latency services from
    # entering the decision solely because their absolute baseline is large.
    if lat_ratio is not None and lat_ratio.ratio is not None and lat_ratio.ratio >= 1.5:
        p90_ms = lat_ratio.post_median * 1000.0
        telemetry["sre"]["p95_latency_ms"] = p90_ms
        telemetry["sre"]["_evidence"]["p95_latency_ms"] = {
            "status": "proxy",
            "source": f"RCAEval {lat_ratio.column}, post-injection 5-minute median p90 latency",
            "note": "conservative p95 lower-bound proxy: observed p90 converted from seconds to milliseconds; gated by >=1.5x pre-injection median",
        }

    # CPU columns are percentages in the observed RCAEval schema. Feed the
    # observed post-injection median only when the same service metric rises by
    # at least 1.5x over its pre-injection baseline.
    if cpu_ratio is not None and cpu_ratio.ratio is not None and cpu_ratio.ratio >= 1.5:
        telemetry["sre"]["saturation_pct"] = cpu_ratio.post_median
        telemetry["sre"]["_evidence"]["saturation_pct"] = {
            "status": "measured",
            "source": f"RCAEval {cpu_ratio.column}, post-injection 5-minute median",
            "note": "system-wide label-blind scan; metric selected by largest post/pre ratio",
        }

    diagnostics = {
        "window_seconds": {"pre": PRE_SECONDS, "post": POST_SECONDS},
        "selection_rule": "system-wide metric scan; RCAEval fault/root-cause labels are not inputs",
        "latency90_largest_ratio": None if lat_ratio is None else lat_ratio.__dict__,
        "latency90_largest_post": None if lat_abs is None else lat_abs.__dict__,
        "cpu_largest_ratio": None if cpu_ratio is None else cpu_ratio.__dict__,
        "cpu_largest_post": None if cpu_abs is None else cpu_abs.__dict__,
        "memory_largest_ratio": None if _largest_ratio(mem) is None else _largest_ratio(mem).__dict__,
        "diskio_largest_ratio": None if _largest_ratio(disk) is None else _largest_ratio(disk).__dict__,
        "socket_largest_ratio": None if _largest_ratio(socket) is None else _largest_ratio(socket).__dict__,
        "error_largest_ratio": None if _largest_ratio(error) is None else _largest_ratio(error).__dict__,
        "unmapped_signals": ["memory", "diskio", "socket", "error"],
        "unmapped_reason": "no direct AAF field with sufficiently verified unit/semantics for this external source; retained only for audit diagnostics",
    }
    return telemetry, diagnostics
