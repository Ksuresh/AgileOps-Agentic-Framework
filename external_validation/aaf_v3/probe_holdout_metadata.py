"""Metadata-only probe of RCAEval RE1/RE3 (PLAN.md: counts, fault types, file layout, column names).
No AAF decisions and no metric values are recorded."""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
from huggingface_hub import HfApi, hf_hub_download

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from external_validation.rcaeval.adapter import adapt  # noqa: E402

REPO = "phamquiluan/RCAEval"
OUT = Path(__file__).resolve().parent / "probe_report"
OUT.mkdir(exist_ok=True)
idx = pd.read_parquet(hf_hub_download(REPO, "cases.parquet", repo_type="dataset"))
files = HfApi().list_repo_files(REPO, repo_type="dataset")
rep = {}
for suite in ("RE1", "RE3"):
    sub = idx[idx["suite"].astype(str) == suite]
    rep[suite] = {"n_cases": len(sub), "by_dataset_fault": {f"{d}|{f}": int(n) for (d, f), n in sub.groupby(["dataset", "fault"]).size().items()},
                  "fault_descriptions": sorted(set(sub["fault_description"].astype(str)))[:20],
                  "has_logs": dict(Counter(sub["has_logs"].astype(str))), "has_traces": dict(Counter(sub["has_traces"].astype(str))), "samples": []}
    for ds in sorted(sub["dataset"].astype(str).unique()):
        for case in sub[sub["dataset"].astype(str) == ds].sort_values("case")["case"].astype(str).head(2):
            fl = sorted(f[len(case) + 1:] for f in files if f.startswith(case + "/"))
            s = {"dataset": ds, "case": case, "files": fl}
            try:
                m = pd.read_parquet(hf_hub_download(REPO, f"{case}/metrics.parquet", repo_type="dataset"))
                cols = list(m.columns)
                suf = Counter(re.sub(r"^.*?_", "_", c) for c in cols if c != "time")
                s.update({"n_columns": len(cols), "has_time": "time" in cols, "column_suffixes_top": dict(suf.most_common(25)),
                          "first_columns": cols[:15]})
                inject = int(hf_hub_download(REPO, f"{case}/inject_time.txt", repo_type="dataset") and open(hf_hub_download(REPO, f"{case}/inject_time.txt", repo_type="dataset")).read().strip())
                try:
                    tel, diag = adapt(m, inject, logs=None)
                    s["adapter_ok"] = True
                    s["adapter_fields_present"] = {k: sorted(v.keys()) if isinstance(v, dict) else None for k, v in tel.items()}
                    s["diag_keys"] = sorted(diag.keys())
                except Exception as exc:
                    s["adapter_ok"] = f"{type(exc).__name__}: {exc}"[:300]
            except Exception as exc:
                s["error"] = f"{type(exc).__name__}: {exc}"[:300]
            rep[suite]["samples"].append(s)
(OUT / "summary.json").write_text(json.dumps(rep, indent=1, default=str))
print(json.dumps({k: {"n": v["n_cases"], "faults": v["by_dataset_fault"]} for k, v in rep.items()}, indent=1))
