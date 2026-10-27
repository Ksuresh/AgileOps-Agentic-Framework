"""Development-only feature export (PLAN.md): RCAEval RE2 fault and control windows, per metric family
(latency-90, cpu, mem, diskio, socket): largest post/pre ratio, robust z (post median vs pre median in units of
pre-window MAD), and counts of metrics over ratio / z cut-offs. Used to design AAF v3 change-aware materiality."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from external_validation.rcaeval_governance.run_study import dev_case_ids  # noqa: E402

REPO = "phamquiluan/RCAEval"
FAM = {"lat": "_latency-90", "cpu": "_cpu", "mem": "_mem", "disk": "_diskio", "sock": "_socket"}
OUT = Path(__file__).resolve().parent / "dev"
OUT.mkdir(exist_ok=True)


def dl(p):
    return Path(hf_hub_download(REPO, p, repo_type="dataset"))


def feats(m: pd.DataFrame, t0: int) -> dict:
    t = pd.to_numeric(m["time"], errors="coerce")
    pre, post = (t >= t0 - 300) & (t < t0), (t >= t0) & (t < t0 + 300)
    out = {}
    for f, suf in FAM.items():
        cols = [c for c in m.columns if c != "time" and c.lower().endswith(suf)]
        best_r, best_z, rows = None, None, []
        for c in cols:
            x = pd.to_numeric(m[c], errors="coerce")
            a, b = x[pre].dropna(), x[post].dropna()
            if a.empty or b.empty:
                continue
            pm, qm = float(a.median()), float(b.median())
            mad = float((a - pm).abs().median()) * 1.4826
            r = None if pm <= 0 else qm / pm
            z = (qm - pm) / mad if mad > 0 else (np.inf if qm > pm else 0.0)
            rows.append((c, pm, qm, r, z))
        rr = [x for x in rows if x[3] is not None]
        if rr:
            c, pm, qm, r, z = max(rr, key=lambda x: x[3])
            out.update({f"{f}_top_ratio": r, f"{f}_top_ratio_col": c, f"{f}_top_ratio_pre": pm, f"{f}_top_ratio_post": qm, f"{f}_top_ratio_z": z})
        fin = [x for x in rows if np.isfinite(x[4])]
        if fin:
            c, pm, qm, r, z = max(fin, key=lambda x: x[4])
            out.update({f"{f}_top_z": z, f"{f}_top_z_col": c, f"{f}_top_z_ratio": r, f"{f}_top_z_post": qm})
        out[f"{f}_n_metrics"] = len(rows)
        for k in (1.5, 2, 3, 5):
            out[f"{f}_n_ratio_ge{k}"] = sum(1 for x in rr if x[3] >= k)
        for k in (3, 5, 10):
            out[f"{f}_n_z_ge{k}"] = sum(1 for x in rows if x[4] >= k)
    return out


idx = pd.read_parquet(dl("cases.parquet"))
re2 = idx[idx["suite"].astype(str) == "RE2"].sort_values("case")
dev = dev_case_ids(idx)
rows = []
for _, row in re2.iterrows():
    case = str(row["case"])
    inject = int(dl(f"{case}/inject_time.txt").read_text().strip())
    m = pd.read_parquet(dl(f"{case}/metrics.parquet"))
    for window, t in (("fault", inject), ("control", inject - 300)):
        rows.append({"case": case, "dataset": str(row["dataset"]), "fault": str(row["fault"]), "window": window,
                     "split": "tt_dev" if case in dev else "heldout", **feats(m, t)})
pd.DataFrame(rows).to_csv(OUT / "re2_window_features.csv", index=False)
print(len(rows))
