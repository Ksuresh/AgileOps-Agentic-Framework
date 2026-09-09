from __future__ import annotations

"""Probe RCAEval metrics/logs/traces so the external adapter can map as much
source evidence as possible without leaking fault labels into AAF.

This script is diagnostic only. It does not call AAF and it does not use the
fault label or root-cause service to select evidence.
"""

from pathlib import Path
import json
import re

import pandas as pd
from huggingface_hub import hf_hub_download

REPO_ID = "phamquiluan/RCAEval"
REPO_TYPE = "dataset"
OUT = Path("paper_results/external_rcaeval/generated/multisource_probe")
PRE_SECONDS = 300
POST_SECONDS = 300
ERROR_RE = re.compile(r"\b(error|exception|fail(?:ed|ure)?|fatal|timeout)\b", re.I)


def dl(path: str) -> Path:
    return Path(hf_hub_download(repo_id=REPO_ID, repo_type=REPO_TYPE, filename=path))


def median_numeric(s: pd.Series) -> float | None:
    x = pd.to_numeric(s, errors="coerce").dropna()
    return None if x.empty else float(x.median())


def ratio(pre: float | None, post: float | None) -> float | None:
    if pre is None or post is None or pre == 0:
        return None
    return post / pre


def count_window(df: pd.DataFrame, tcol: str, inject_time: int, mask: pd.Series | None = None) -> dict:
    t = pd.to_numeric(df[tcol], errors="coerce")
    pre = (t >= inject_time - PRE_SECONDS) & (t < inject_time)
    post = (t >= inject_time) & (t < inject_time + POST_SECONDS)
    if mask is not None:
        pre &= mask
        post &= mask
    a, b = int(pre.sum()), int(post.sum())
    return {"pre_5m": a, "post_5m": b, "ratio": None if a == 0 else b / a}


def profile_case(case: str, inject_time: int) -> dict:
    metrics = pd.read_parquet(dl(f"{case}/metrics.parquet"))
    t = pd.to_numeric(metrics["time"], errors="coerce")
    pre_mask = (t >= inject_time - PRE_SECONDS) & (t < inject_time)
    post_mask = (t >= inject_time) & (t < inject_time + POST_SECONDS)

    families: dict[str, list[dict]] = {}
    for suffix in ("_mem", "_diskio", "_socket", "_cpu", "_latency-90", "_latency-50"):
        vals = []
        for col in metrics.columns:
            if not col.lower().endswith(suffix):
                continue
            pre = median_numeric(metrics.loc[pre_mask, col])
            post = median_numeric(metrics.loc[post_mask, col])
            vals.append({"column": col, "pre_median": pre, "post_median": post, "ratio": ratio(pre, post)})
        vals.sort(key=lambda x: float(x["ratio"] or -1), reverse=True)
        families[suffix] = vals[:10]

    trace_info: dict = {"available": False}
    try:
        traces = pd.read_parquet(dl(f"{case}/traces.parquet"))
        trace_info = {"available": True, "rows": int(len(traces)), "columns": list(map(str, traces.columns))}
        time_col = next((c for c in ("time", "startTimeMillis", "startTime") if c in traces.columns), None)
        if time_col:
            tt = pd.to_numeric(traces[time_col], errors="coerce")
            # Normalize millisecond timestamps if required.
            if tt.dropna().median() and tt.dropna().median() > 10_000_000_000:
                normalized = tt / 1000.0
            else:
                normalized = tt
            temp = traces.copy()
            temp["__time_s"] = normalized
            trace_info["span_counts"] = count_window(temp, "__time_s", inject_time)
            if "duration" in temp.columns:
                pre = median_numeric(temp.loc[(normalized >= inject_time-PRE_SECONDS) & (normalized < inject_time), "duration"])
                post = median_numeric(temp.loc[(normalized >= inject_time) & (normalized < inject_time+POST_SECONDS), "duration"])
                trace_info["duration"] = {"pre_median": pre, "post_median": post, "ratio": ratio(pre, post)}
            if "statusCode" in temp.columns:
                sc = temp["statusCode"]
                trace_info["status_non_null"] = int(sc.notna().sum())
                trace_info["status_values"] = {str(k): int(v) for k, v in sc.dropna().astype(str).value_counts().head(20).items()}
                errmask = sc.notna() & ~sc.astype(str).str.lower().isin({"0", "ok", "unset", "none", "nan", "success"})
                trace_info["non_ok_status_counts"] = count_window(temp, "__time_s", inject_time, errmask)
    except Exception as exc:
        trace_info = {"available": False, "reason": type(exc).__name__}

    log_info: dict = {"available": False}
    try:
        logs = pd.read_parquet(dl(f"{case}/logs.parquet"))
        log_info = {"available": True, "rows": int(len(logs)), "columns": list(map(str, logs.columns))}
        tcol = next((c for c in ("timestamp", "time") if c in logs.columns), None)
        mcol = next((c for c in ("message", "log", "body") if c in logs.columns), None)
        if tcol and mcol:
            lt = pd.to_numeric(logs[tcol], errors="coerce")
            if lt.dropna().median() and lt.dropna().median() > 10_000_000_000:
                normalized = lt / 1000.0
            else:
                normalized = lt
            temp = logs.copy()
            temp["__time_s"] = normalized
            msgs = temp[mcol].fillna("").astype(str)
            errmask = msgs.str.contains(ERROR_RE, regex=True)
            log_info["all_log_counts"] = count_window(temp, "__time_s", inject_time)
            log_info["error_like_counts"] = count_window(temp, "__time_s", inject_time, errmask)
    except Exception as exc:
        log_info = {"available": False, "reason": type(exc).__name__}

    return {"case": case, "metric_families": families, "traces": trace_info, "logs": log_info}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    idx = pd.read_parquet(dl("cases.parquet"))
    subset = idx[idx["dataset"].astype(str).str.upper() == "RE2-TT"].copy()
    selected = []
    for fault in sorted(subset["fault"].astype(str).unique()):
        rows = subset[subset["fault"].astype(str) == fault].sort_values("case").head(2)
        for _, row in rows.iterrows():
            selected.append((str(row["case"]), int(row["inject_time"])))

    payload = {
        "dataset": "RCAEval RE2-TT",
        "purpose": "adapter semantics verification only",
        "selection": "same 12 pilot cases; two per fault class",
        "fault_labels_used_for_decision": False,
        "cases": [profile_case(case, inject_time) for case, inject_time in selected],
    }
    (OUT / "multisource_probe.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({
        "cases": len(payload["cases"]),
        "trace_available": sum(1 for x in payload["cases"] if x["traces"].get("available")),
        "logs_available": sum(1 for x in payload["cases"] if x["logs"].get("available")),
    }, indent=2))


if __name__ == "__main__":
    main()
