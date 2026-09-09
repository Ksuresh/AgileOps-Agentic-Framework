from __future__ import annotations

"""Inspect RCAEval RE2-TT source telemetry before freezing the AAF adapter."""

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


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    idx = pd.read_parquet(dl("cases.parquet"))
    subset = idx[idx["dataset"].astype(str).str.upper() == "RE2-TT"].copy()
    if subset.empty:
        raise RuntimeError("RCAEval index contains no RE2-TT cases")

    row = subset.iloc[0]
    case = str(row["case"])
    metrics = pd.read_parquet(dl(f"{case}/metrics.parquet"))
    traces = pd.read_parquet(dl(f"{case}/traces.parquet"))
    inject_time = int(dl(f"{case}/inject_time.txt").read_text(encoding="utf-8").strip())

    metric_cols = [c for c in metrics.columns if c != "time"]
    suffixes: dict[str, int] = {}
    for col in metric_cols:
        suffix = col.rsplit("_", 1)[-1].lower() if "_" in col else "other"
        suffixes[suffix] = suffixes.get(suffix, 0) + 1

    trace_summary = {"columns": list(map(str, traces.columns)), "rows": int(len(traces))}
    for col in ("time", "startTimeMillis", "startTime", "duration", "statusCode", "serviceName"):
        if col not in traces.columns:
            continue
        s = traces[col]
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
        "sample_case": case,
        "ground_truth_service": str(row.get("root_cause_service", "")),
        "fault_label": str(row.get("fault", "")),
        "inject_time": inject_time,
        "metrics": {
            "rows": int(len(metrics)),
            "columns": int(len(metrics.columns)),
            "time_dtype": str(metrics["time"].dtype) if "time" in metrics.columns else None,
            "time_range": describe_numeric(metrics["time"]) if "time" in metrics.columns else {},
            "metric_suffix_counts": dict(sorted(suffixes.items())),
            "columns": metric_cols,
        },
        "traces": trace_summary,
        "decision_input_warning": "fault_label and ground_truth_service are audit metadata only and must not enter AAF",
    }

    out = OUT / "schema_probe.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
