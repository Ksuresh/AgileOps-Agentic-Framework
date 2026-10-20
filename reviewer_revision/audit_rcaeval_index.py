"""Acquire RCAEval's public case index and audit metadata without downloading telemetry.

Runs in GitHub Actions with network access. This is a discovery audit only:
root-cause labels are not governance-action labels.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen
from datetime import datetime, timezone

import pandas as pd

URL = "https://huggingface.co/datasets/phamquiluan/RCAEval/resolve/main/cases.parquet"
OUT = Path(__file__).resolve().parent / "results" / "external_dataset_audit"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with urlopen(URL, timeout=120) as response:
        raw = response.read()
    digest = hashlib.sha256(raw).hexdigest()
    source = OUT / "rcaeval_cases.parquet"
    source.write_bytes(raw)
    frame = pd.read_parquet(source)
    required = {
        "case", "dataset", "suite", "system", "root_cause_service",
        "fault", "repetition", "inject_time", "has_logs", "has_traces",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError("Unexpected RCAEval schema: " + ", ".join(missing))
    if frame["case"].duplicated().any():
        raise ValueError("Duplicate RCAEval case identifiers")
    report = {
        "status": "external_metadata_audit_not_governance_ground_truth",
        "source": URL,
        "retrieved_utc": datetime.now(timezone.utc).isoformat(),
        "sha256": digest,
        "index_bytes": len(raw),
        "rows": len(frame),
        "columns": list(frame.columns),
        "null_counts": {str(k): int(v) for k, v in frame.isna().sum().items()},
        "by_dataset": {str(k): int(v) for k, v in frame.groupby("dataset").size().items()},
        "by_suite": {str(k): int(v) for k, v in frame.groupby("suite").size().items()},
        "by_system": {str(k): int(v) for k, v in frame.groupby("system").size().items()},
        "with_logs": int(frame["has_logs"].sum()),
        "with_traces": int(frame["has_traces"].sum()),
        "with_both": int((frame["has_logs"] & frame["has_traces"]).sum()),
        "unique_faults": sorted(frame["fault"].dropna().astype(str).unique().tolist()),
        "unique_root_cause_services": int(frame["root_cause_service"].nunique()),
        "caution": "Dataset contains root-cause labels, NOT validated governance-action labels.",
    }
    (OUT / "rcaeval_metadata_audit.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
