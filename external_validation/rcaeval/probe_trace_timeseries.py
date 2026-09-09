from __future__ import annotations

"""Probe RCAEval trace-derived time-series on the frozen 12 pilot cases.

This diagnostic uses no fault label or root-cause service as AAF input. It only
verifies source-file availability, schema, and pre/post behavior before the
external evidence adapter is frozen.
"""

from pathlib import Path
import json

import pandas as pd
from huggingface_hub import hf_hub_download

REPO_ID = "phamquiluan/RCAEval"
REPO_TYPE = "dataset"
OUT = Path("paper_results/external_rcaeval/generated/trace_ts_probe")
PRE_SECONDS = 300
POST_SECONDS = 300


def dl(path: str) -> Path:
    return Path(hf_hub_download(repo_id=REPO_ID, repo_type=REPO_TYPE, filename=path))


def read_optional(case: str, filename: str) -> tuple[pd.DataFrame | None, str | None]:
    candidates = [f"{case}/{filename}", f"{case}/{filename.replace('.csv', '.parquet')}"]
    for path in candidates:
        try:
            p = dl(path)
            if p.suffix == ".parquet":
                return pd.read_parquet(p), path
            return pd.read_csv(p), path
        except Exception:
            pass
    return None, None


def summarize(df: pd.DataFrame, inject_time: int) -> dict:
    if df is None or df.empty:
        return {"available": False}
    time_col = next((c for c in ("time", "timestamp", "ts") if c in df.columns), None)
    out = {"available": True, "rows": int(len(df)), "columns": list(map(str, df.columns)), "time_col": time_col}
    if not time_col:
        return out
    t = pd.to_numeric(df[time_col], errors="coerce")
    if t.dropna().empty:
        return out
    if float(t.dropna().median()) > 10_000_000_000:
        t = t / 1000.0
    pre_mask = (t >= inject_time - PRE_SECONDS) & (t < inject_time)
    post_mask = (t >= inject_time) & (t < inject_time + POST_SECONDS)
    signals = []
    for c in df.columns:
        if c == time_col:
            continue
        x = pd.to_numeric(df[c], errors="coerce")
        pre = x[pre_mask].dropna()
        post = x[post_mask].dropna()
        if pre.empty or post.empty:
            continue
        pre_med = float(pre.median())
        post_med = float(post.median())
        signals.append({
            "column": str(c),
            "pre_median": pre_med,
            "post_median": post_med,
            "ratio": None if pre_med == 0 else post_med / pre_med,
            "pre_sum": float(pre.sum()),
            "post_sum": float(post.sum()),
        })
    signals.sort(key=lambda x: float(x["ratio"] if x["ratio"] is not None else -1), reverse=True)
    out["top_ratio_signals"] = signals[:20]
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    idx = pd.read_parquet(dl("cases.parquet"))
    subset = idx[idx["dataset"].astype(str).str.upper() == "RE2-TT"].copy()
    cases = []
    for fault in sorted(subset["fault"].astype(str).unique()):
        rows = subset[subset["fault"].astype(str) == fault].sort_values("case").head(2)
        for _, row in rows.iterrows():
            case = str(row["case"])
            inject_time = int(row["inject_time"])
            err, err_path = read_optional(case, "tracets_err.csv")
            lat, lat_path = read_optional(case, "tracets_lat.csv")
            logts, log_path = read_optional(case, "logts.csv")
            cases.append({
                "case": case,
                "fault_label_audit_only": str(row["fault"]),
                "inject_time": inject_time,
                "tracets_err_path": err_path,
                "tracets_err": summarize(err, inject_time) if err is not None else {"available": False},
                "tracets_lat_path": lat_path,
                "tracets_lat": summarize(lat, inject_time) if lat is not None else {"available": False},
                "logts_path": log_path,
                "logts": summarize(logts, inject_time) if logts is not None else {"available": False},
            })
    payload = {
        "dataset": "RCAEval RE2-TT",
        "purpose": "final external-adapter semantics verification",
        "pilot_cases": 12,
        "labels_used_by_aaf": False,
        "cases": cases,
    }
    (OUT / "trace_timeseries_probe.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({
        "cases": len(cases),
        "tracets_err_available": sum(1 for c in cases if c["tracets_err"].get("available")),
        "tracets_lat_available": sum(1 for c in cases if c["tracets_lat"].get("available")),
        "logts_available": sum(1 for c in cases if c["logts"].get("available")),
    }, indent=2))


if __name__ == "__main__":
    main()
