from __future__ import annotations

"""Inspect one RCAEval RE2-TrainTicket case before freezing the evidence adapter.

This probe intentionally does not call AAF. Its sole purpose is to document the
available external telemetry fields, timestamp units, duration/status ranges,
and metric naming so that the later AAF adapter is based on observed schema
rather than assumptions.
"""

from pathlib import Path
import json

import pandas as pd
from huggingface_hub import hf_hub_download

REPO_ID = "phamquiluan/RCAEval"
REPO_TYPE = "dataset"
OUT = Path("results_external_rcaeval_probe")


def dl(path: str) -> Path:
    return Path(hf_hub_download(repo_id=REPO_ID, repo_type=REPO_TYPE, filename=path))


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
    interesting_metric_cols = [
        c for c in metric_cols
        if any(k in c.lower() for k in ("lat", "duration", "response", "cpu", "mem", "disk", "error", "success", "request"))
    ]

    trace_summary = {
        "columns": list(map(str, traces.columns)),
        "rows": int(len(traces)),
    }
    for col in ("time", "startTime", "duration", "statusCode", "serviceName"):
        if col in traces.columns:
            s = traces[col]
            info = {
                "dtype": str(s.dtype),
                "non_null": int(s.notna().sum()),
                "sample": [None if pd.isna(x) else str(x) for x in s.dropna().head(10).tolist()],
            }
            if pd.api.types.is_numeric_dtype(s):
                vals = pd.to_numeric(s, errors="coerce").dropna()
                if not vals.empty:
                    info.update({
                        "min": float(vals.min()),
                        "median": float(vals.median()),
                        "p95": float(vals.quantile(0.95)),
                        "max": float(vals.max()),
                    })
            trace_summary[col] = info

    payload = {
        "dataset": "RCAEval RE2-TT",
        "case_count": int(len(subset)),
        "sample_case": case,
        "ground_truth_service": str(row.get("root_cause_service", "")),
        "fault_label": str(row.get("fault", "")),
        "inject_time": inject_time,
        "metrics": {
            "rows": int(len(metrics)),
            "columns": int(len(metrics.columns)),
            "time_dtype": str(metrics["time"].dtype) if "time" in metrics.columns else None,
            "time_min": float(pd.to_numeric(metrics["time"], errors="coerce").min()) if "time" in metrics.columns else None,
            "time_max": float(pd.to_numeric(metrics["time"], errors="coerce").max()) if "time" in metrics.columns else None,
            "interesting_columns": interesting_metric_cols[:120],
        },
        "traces": trace_summary,
    }

    out = OUT / "schema_probe.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
