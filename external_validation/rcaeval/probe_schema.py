from __future__ import annotations

"""Profile RCAEval RE2-TrainTicket telemetry before freezing the AAF adapter.

The profiler samples one case from each fault class and reports only source
schema/statistics. It never calls AAF and never uses fault labels or root-cause
metadata as model input.
"""

from pathlib import Path
import json

import pandas as pd
from huggingface_hub import hf_hub_download

REPO_ID = "phamquiluan/RCAEval"
REPO_TYPE = "dataset"
OUT = Path("paper_results/external_rcaeval/generated/probe")


def dl(path: str) -> Path:
    return Path(hf_hub_download(repo_id=REPO_ID, repo_type=REPO_TYPE, filename=path))


def describe_numeric(s: pd.Series) -> dict:
    vals = pd.to_numeric(s, errors="coerce").dropna()
    if vals.empty:
        return {}
    return {
        "min": float(vals.min()),
        "median": float(vals.median()),
        "p95": float(vals.quantile(0.95)),
        "max": float(vals.max()),
    }


def window_stats(metrics: pd.DataFrame, inject_time: int, col: str) -> dict:
    if "time" not in metrics.columns or col not in metrics.columns:
        return {}
    t = pd.to_numeric(metrics["time"], errors="coerce")
    x = pd.to_numeric(metrics[col], errors="coerce")
    pre = x[(t >= inject_time - 300) & (t < inject_time)]
    post = x[(t >= inject_time) & (t < inject_time + 300)]
    result = {"column": col, "pre_5m": describe_numeric(pre), "post_5m": describe_numeric(post)}
    if pre.dropna().size and post.dropna().size:
        pre_med = float(pre.median())
        post_med = float(post.median())
        result["median_delta"] = post_med - pre_med
        result["median_ratio"] = None if pre_med == 0 else post_med / pre_med
    return result


def profile_case(row: pd.Series) -> dict:
    case = str(row["case"])
    root = str(row.get("root_cause_service", ""))
    fault = str(row.get("fault", ""))
    metrics = pd.read_parquet(dl(f"{case}/metrics.parquet"))
    inject_time = int(dl(f"{case}/inject_time.txt").read_text(encoding="utf-8").strip())

    root_cols = [c for c in metrics.columns if c != "time" and c.startswith(root + "_")]
    cpu_col = next((c for c in root_cols if c.lower().endswith("_cpu")), None)
    mem_col = next((c for c in root_cols if c.lower().endswith("_mem")), None)
    latency_cols = [c for c in root_cols if any(k in c.lower() for k in ("latency", "duration", "response"))]

    return {
        "case": case,
        "fault_label": fault,
        "ground_truth_service": root,
        "inject_time": inject_time,
        "metric_rows": int(len(metrics)),
        "metric_columns": int(len(metrics.columns)),
        "root_service_columns": root_cols,
        "cpu": window_stats(metrics, inject_time, cpu_col) if cpu_col else {},
        "mem": window_stats(metrics, inject_time, mem_col) if mem_col else {},
        "root_latency_candidates": [window_stats(metrics, inject_time, c) for c in latency_cols[:20]],
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    idx = pd.read_parquet(dl("cases.parquet"))
    subset = idx[idx["dataset"].astype(str).str.upper() == "RE2-TT"].copy()
    if subset.empty:
        raise RuntimeError("RCAEval index contains no RE2-TT cases")

    representatives = []
    for fault in sorted(subset["fault"].astype(str).unique()):
        representatives.append(profile_case(subset[subset["fault"].astype(str) == fault].iloc[0]))

    first = subset.iloc[0]
    sample_case = str(first["case"])
    sample_metrics = pd.read_parquet(dl(f"{sample_case}/metrics.parquet"))
    sample_traces = pd.read_parquet(dl(f"{sample_case}/traces.parquet"))

    metric_cols = [c for c in sample_metrics.columns if c != "time"]
    suffixes: dict[str, int] = {}
    for col in metric_cols:
        suffix = col.rsplit("_", 1)[-1].lower() if "_" in col else "other"
        suffixes[suffix] = suffixes.get(suffix, 0) + 1

    trace_summary = {"columns": list(map(str, sample_traces.columns)), "rows": int(len(sample_traces))}
    for col in ("time", "startTimeMillis", "startTime", "duration", "statusCode", "serviceName"):
        if col not in sample_traces.columns:
            continue
        s = sample_traces[col]
        info = {
            "dtype": str(s.dtype),
            "non_null": int(s.notna().sum()),
            "sample": [str(x) for x in s.dropna().head(10).tolist()],
        }
        if pd.api.types.is_numeric_dtype(s):
            info.update(describe_numeric(s))
        trace_summary[col] = info

    payload = {
        "dataset": "RCAEval RE2-TT",
        "case_count": int(len(subset)),
        "fault_counts": {str(k): int(v) for k, v in subset["fault"].value_counts().sort_index().items()},
        "sample_schema": {
            "case": sample_case,
            "metric_rows": int(len(sample_metrics)),
            "metric_columns": int(len(sample_metrics.columns)),
            "metric_suffix_counts": dict(sorted(suffixes.items())),
            "traces": trace_summary,
        },
        "representative_fault_profiles": representatives,
        "decision_input_warning": "fault_label and ground_truth_service are audit metadata only and must not enter AAF",
    }

    out = OUT / "schema_probe.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
