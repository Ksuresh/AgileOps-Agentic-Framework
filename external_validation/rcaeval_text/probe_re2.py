"""Schema probe for RCAEval RE2 (all three systems) to design the text-evidence study.

Read-only. Writes a small markdown/JSON report: case index structure, label columns,
metric column families, log/trace schema, line counts around injection and a few
example log lines per sampled case. No full telemetry is written or committed.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import RemoteEntryNotFoundError

REPO_ID = "phamquiluan/RCAEval"
OUT = Path(__file__).resolve().parent / "probe_report"
WINDOW = 300


def dl(path: str) -> Path:
    return Path(hf_hub_download(repo_id=REPO_ID, repo_type="dataset", filename=path))


def maybe(path: str) -> pd.DataFrame | None:
    try:
        return pd.read_parquet(dl(path))
    except (RemoteEntryNotFoundError, FileNotFoundError):
        return None
    except Exception as exc:  # report, do not crash the probe
        return pd.DataFrame({"_probe_error": [f"{type(exc).__name__}: {exc}"]})


def family(col: str) -> str:
    return re.sub(r"^.*?_", "", col) if "_" in col else col


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    idx = pd.read_parquet(dl("cases.parquet"))
    report: dict = {"index_columns": idx.columns.tolist(), "index_rows": len(idx),
                    "datasets": idx["dataset"].astype(str).value_counts().to_dict()}
    re2 = idx[idx["dataset"].astype(str).str.upper().str.startswith("RE2")].copy()
    report["re2_rows"] = len(re2)
    for col in re2.columns:
        if re2[col].nunique() <= 30:
            report.setdefault("re2_small_columns", {})[col] = re2[col].astype(str).value_counts().to_dict()
    report["re2_head"] = re2.head(5).astype(str).to_dict(orient="records")

    sample = (re2.sort_values("case").groupby(["dataset", "fault"], as_index=False).head(1))
    lines = ["# RCAEval RE2 probe", "", f"Index rows: {len(idx)}; RE2 rows: {len(re2)}", "",
             "Index columns: " + ", ".join(map(str, idx.columns)), ""]
    cases = []
    for _, row in sample.iterrows():
        case = str(row["case"])
        rec = {"case": case, "dataset": str(row["dataset"]), "fault": str(row["fault"]),
               "labels": {k: str(v) for k, v in row.items() if k not in ("case",)}}
        try:
            inject = int(dl(f"{case}/inject_time.txt").read_text().strip())
        except Exception as exc:
            inject = None
            rec["inject_error"] = str(exc)
        rec["inject_time"] = inject
        metrics = maybe(f"{case}/metrics.parquet")
        if metrics is not None:
            cols = [c for c in metrics.columns if c != "time"]
            rec["metric_columns_n"] = len(cols)
            rec["metric_families"] = dict(Counter(family(c) for c in cols).most_common(25))
            rec["metric_examples"] = cols[:15]
        logs = maybe(f"{case}/logs.parquet")
        if logs is not None and "_probe_error" not in logs.columns:
            rec["log_columns"] = logs.columns.tolist()
            rec["log_rows"] = len(logs)
            tcol = next((c for c in ("timestamp", "time") if c in logs.columns), None)
            mcol = next((c for c in ("message", "log", "body") if c in logs.columns), None)
            scol = next((c for c in ("container_name", "service", "pod", "container") if c in logs.columns), None)
            if tcol and mcol and inject is not None:
                t = pd.to_numeric(logs[tcol], errors="coerce")
                if not t.dropna().empty and float(t.dropna().median()) > 10_000_000_000:
                    t = t / (1000.0 if float(t.dropna().median()) < 1e14 else 1e9)
                post = logs[(t >= inject) & (t < inject + WINDOW)]
                pre = logs[(t >= inject - WINDOW) & (t < inject)]
                rec["log_lines_pre_post"] = [len(pre), len(post)]
                if scol:
                    rec["log_services_post_top"] = post[scol].astype(str).value_counts().head(8).to_dict()
                msgs = post[mcol].astype(str)
                err = msgs[msgs.str.contains(r"error|exception|fail|timeout|refused|restart|killed|oom", case=False, regex=True)]
                rec["post_error_like_lines"] = int(len(err))
                rec["example_post_lines"] = [m[:220] for m in err.head(6).tolist()] or [m[:220] for m in msgs.head(4).tolist()]
        elif logs is not None:
            rec["logs_error"] = logs["_probe_error"].iloc[0]
        traces = maybe(f"{case}/traces.parquet")
        if traces is not None and "_probe_error" not in traces.columns:
            rec["trace_columns"] = traces.columns.tolist()
            rec["trace_rows"] = len(traces)
        cases.append(rec)
        lines += [f"## {case} ({rec['dataset']}, {rec['fault']})", "",
                  "```json", json.dumps({k: v for k, v in rec.items() if k != "labels"}, indent=1, default=str)[:4000], "```", ""]
    report["sampled_cases"] = cases
    (OUT / "probe_re2.json").write_text(json.dumps(report, indent=1, default=str))
    (OUT / "probe_re2.md").write_text("\n".join(lines))
    print("\n".join(lines[:40]))


if __name__ == "__main__":
    main()
