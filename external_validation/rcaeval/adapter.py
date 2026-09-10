from __future__ import annotations

"""Label-blind RCAEval -> AAF evidence adapter.

Decision-active mapping is intentionally narrow and frozen around AAF fields
whose semantics can be defended directly from RCAEval: service p90 latency as a
conservative lower-bound proxy for AAF p95 latency, and CPU percentage as AAF
resource saturation. Additional RCAEval observables (memory, disk I/O, socket
activity, and log activity) are retained as measured auxiliary evidence so the
external validation does not discard source information, but they are not
silently coerced into unrelated AAF decision fields.

RCAEval fault labels and root-cause service annotations are never inputs.
"""

from dataclasses import dataclass
from typing import Any
import re

import pandas as pd

PRE_SECONDS = 300
POST_SECONDS = 300
RATIO_GATE = 1.5
ERROR_RE = re.compile(r"\b(?:error|exception|fail(?:ed|ure)?|fatal|timeout)\b", re.I)


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


def _as_dict(change: Change | None) -> dict[str, Any] | None:
    return None if change is None else change.__dict__


def _log_summary(logs: pd.DataFrame | None, inject_time: int) -> dict[str, Any]:
    if logs is None or logs.empty:
        return {"available": False}
    tcol = next((c for c in ("timestamp", "time") if c in logs.columns), None)
    mcol = next((c for c in ("message", "log", "body") if c in logs.columns), None)
    if not tcol or not mcol:
        return {"available": False, "reason": "timestamp/message fields unavailable"}
    t = pd.to_numeric(logs[tcol], errors="coerce")
    if not t.dropna().empty and float(t.dropna().median()) > 10_000_000_000:
        t = t / 1000.0
    pre = (t >= inject_time - PRE_SECONDS) & (t < inject_time)
    post = (t >= inject_time) & (t < inject_time + POST_SECONDS)
    msg = logs[mcol].fillna("").astype(str)
    err = msg.str.contains(ERROR_RE, regex=True)
    pre_total, post_total = int(pre.sum()), int(post.sum())
    pre_err, post_err = int((pre & err).sum()), int((post & err).sum())
    pre_density = None if pre_total == 0 else 100.0 * pre_err / pre_total
    post_density = None if post_total == 0 else 100.0 * post_err / post_total
    return {
        "available": True,
        "pre_log_lines": pre_total,
        "post_log_lines": post_total,
        "pre_error_like_lines": pre_err,
        "post_error_like_lines": post_err,
        "pre_error_like_density_pct": pre_density,
        "post_error_like_density_pct": post_density,
        "error_like_count_ratio": None if pre_err == 0 else post_err / pre_err,
        "decision_active": False,
        "reason_not_decision_active": "error-like log density is not equivalent to AAF HTTP error_rate_pct",
    }


def adapt(
    metrics: pd.DataFrame,
    inject_time: int,
    logs: pd.DataFrame | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return (AAF telemetry, adapter diagnostics) without label/oracle input."""
    if "time" not in metrics.columns:
        raise ValueError("RCAEval metrics require a time column")

    latency90 = _changes(metrics, inject_time, "_latency-90")
    cpu = _changes(metrics, inject_time, "_cpu")
    mem = _changes(metrics, inject_time, "_mem")
    disk = _changes(metrics, inject_time, "_diskio")
    socket = _changes(metrics, inject_time, "_socket")

    lat_ratio = _largest_ratio(latency90)
    lat_abs = _largest_post(latency90)
    cpu_ratio = _largest_ratio(cpu)
    cpu_abs = _largest_post(cpu)
    mem_ratio = _largest_ratio(mem)
    disk_ratio = _largest_ratio(disk)
    socket_ratio = _largest_ratio(socket)
    logs_diag = _log_summary(logs, inject_time)

    telemetry: dict[str, Any] = {
        "deploy": {"_evidence": {}},
        "sre": {
            "_evidence": {},
            "_auxiliary": {
                "memory": _as_dict(mem_ratio),
                "diskio": _as_dict(disk_ratio),
                "socket": _as_dict(socket_ratio),
                "logs": logs_diag,
            },
        },
        "finops": {"_evidence": {}},
        "sec": {"_evidence": {}},
    }

    # RCAEval exposes service latency-90, not p95. Because p95 >= p90, a measured
    # p90 crossing an AAF p95 threshold is a conservative lower-bound proxy. A
    # >=1.5x pre/post gate avoids feeding unrelated high-baseline services.
    if lat_ratio is not None and lat_ratio.ratio is not None and lat_ratio.ratio >= RATIO_GATE:
        p90_ms = lat_ratio.post_median * 1000.0
        telemetry["sre"]["p95_latency_ms"] = p90_ms
        telemetry["sre"]["_evidence"]["p95_latency_ms"] = {
            "status": "proxy",
            "source": f"RCAEval {lat_ratio.column}, post-injection 5-minute median p90 latency",
            "note": "conservative p95 lower-bound proxy: observed p90 converted from seconds to milliseconds; gated by >=1.5x pre-injection median",
        }

    # CPU columns are percentages in the observed RCAEval schema. Feed the
    # measured post-injection median only when the same metric rises >=1.5x.
    if cpu_ratio is not None and cpu_ratio.ratio is not None and cpu_ratio.ratio >= RATIO_GATE:
        telemetry["sre"]["saturation_pct"] = cpu_ratio.post_median
        telemetry["sre"]["_evidence"]["saturation_pct"] = {
            "status": "measured",
            "source": f"RCAEval {cpu_ratio.column}, post-injection 5-minute median",
            "note": "system-wide label-blind scan; metric selected by largest post/pre ratio",
        }

    diagnostics = {
        "adapter_policy": "final-pilot candidate",
        "window_seconds": {"pre": PRE_SECONDS, "post": POST_SECONDS},
        "ratio_gate": RATIO_GATE,
        "selection_rule": "system-wide metric scan; RCAEval fault/root-cause labels are not inputs",
        "decision_active_fields": ["p95_latency_ms (from p90 lower-bound proxy)", "saturation_pct (CPU percent)"],
        "latency90_largest_ratio": _as_dict(lat_ratio),
        "latency90_largest_post": _as_dict(lat_abs),
        "cpu_largest_ratio": _as_dict(cpu_ratio),
        "cpu_largest_post": _as_dict(cpu_abs),
        "auxiliary_evidence": {
            "memory_largest_ratio": _as_dict(mem_ratio),
            "diskio_largest_ratio": _as_dict(disk_ratio),
            "socket_largest_ratio": _as_dict(socket_ratio),
            "logs": logs_diag,
        },
        "auxiliary_policy": "retained and reported but not coerced into unrelated AAF fields",
    }
    return telemetry, diagnostics
