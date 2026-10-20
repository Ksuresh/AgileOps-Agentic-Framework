"""Inspect a small, bounded sample of RCAEval telemetry Parquet files.

Uses Hugging Face's repository API for actual per-case paths, not case index
filenames guessed from case IDs. Only reads a small number of Parquet files
and emits aggregate schema/row counts, never raw log contents.
"""
from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen
import pandas as pd

BASE = "https://huggingface.co/api/datasets/phamquiluan/RCAEval/tree/main"
OUT = Path(__file__).resolve().parent / "results/external_dataset_audit"
MAX_FILES = 3
MAX_BYTES = 30_000_000

def request(url):
    return urlopen(Request(url, headers={"User-Agent": "ieee-access-research-audit/1.0"}), timeout=90)

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with request(BASE + "?recursive=true&limit=1000") as response:
        entries = json.load(response)
    # Candidate files must be actual repo paths, not root-cause oracle fields.
    candidates = sorted(
        (x for x in entries if x.get("type") == "file"
         and str(x.get("path", "")).lower().startswith("re2")
         and str(x.get("path", "")).endswith(".parquet")
         and 0 < int(x.get("size") or 0) <= MAX_BYTES),
        key=lambda x: str(x["path"]),
    )
    sampled = []
    for entry in candidates[:MAX_FILES]:
        path = entry["path"]
        url = "https://huggingface.co/datasets/phamquiluan/RCAEval/resolve/main/" + quote(path, safe="/")
        with request(url) as response:
            raw = response.read(MAX_BYTES + 1)
        if len(raw) > MAX_BYTES:
            raise ValueError("Unexpected oversized Parquet file: " + path)
        import io
        frame = pd.read_parquet(io.BytesIO(raw))
        sampled.append({"path": path, "bytes": len(raw), "rows": len(frame),
                        "columns": list(map(str, frame.columns)),
                        "missing_cells": int(frame.isna().sum().sum())})
    report = {"status": "raw_telemetry_sample_verified" if sampled else "no_eligible_paths_found",
              "retrieved_utc": datetime.now(timezone.utc).isoformat(),
              "candidate_paths_in_first_page": len(candidates),
              "sampled": sampled,
              "limitation": "Bounded sample only; no governance labels or held-out eligibility confirmed."}
    (OUT / "rcaeval_telemetry_sample.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if not sampled:
        raise RuntimeError("No RCAEval RE2 parquet candidates found; inspect repository tree layout")

if __name__ == "__main__":
    main()
