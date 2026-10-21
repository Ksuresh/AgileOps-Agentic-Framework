"""Second read-only probe: does the fault show up in the logs, or only in metrics?

For one case per (system, fault) in RCAEval RE2, compare the 5 minutes before injection with
the 5 minutes after: error-like log lines (all services and the annotated root-cause service),
messages that are new after injection, and error / latency-90 / cpu / mem metrics of the
root-cause service. Writes a small report only.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import RemoteEntryNotFoundError

REPO_ID = "phamquiluan/RCAEval"
OUT = Path(__file__).resolve().parent / "probe_report"
W = 300
ERR = re.compile(r"error|exception|fail|timeout|refused|unavailable|reset|killed|oom|deadline", re.I)
NUMS = re.compile(r"\d+")


def dl(p: str) -> Path:
    return Path(hf_hub_download(repo_id=REPO_ID, repo_type="dataset", filename=p))


def seconds(s: pd.Series) -> pd.Series:
    t = pd.to_numeric(s, errors="coerce")
    med = float(t.dropna().median()) if not t.dropna().empty else 0
    if med > 1e14:
        return t / 1e9
    if med > 1e11:
        return t / 1e3
    return t


def template(m: str) -> str:
    return NUMS.sub("#", re.sub(r"[0-9a-f]{8}-[0-9a-f-]{27,}", "<id>", m))[:160]


def med(df: pd.DataFrame, col: str, mask) -> float | None:
    x = pd.to_numeric(df.loc[mask, col], errors="coerce").dropna()
    return None if x.empty else round(float(x.median()), 4)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    idx = pd.read_parquet(dl("cases.parquet"))
    re2 = idx[idx["suite"].astype(str) == "RE2"]
    sample = re2.sort_values("case").groupby(["dataset", "fault"], as_index=False).head(1)
    out = []
    for _, row in sample.iterrows():
        case, rc, inject = str(row["case"]), str(row["root_cause_service"]), int(row["inject_time"])
        rec = {"case": case, "dataset": row["dataset"], "fault": row["fault"], "root_cause": rc}
        m = pd.read_parquet(dl(f"{case}/metrics.parquet"))
        t = seconds(m["time"])
        pre, post = (t >= inject - W) & (t < inject), (t >= inject) & (t < inject + W)
        rc_cols = [c for c in m.columns if c.startswith(rc + "_") or c.startswith(rc.replace("service", "") + "_")]
        rec["rc_metrics"] = {c: [med(m, c, pre), med(m, c, post)] for c in rc_cols
                             if any(k in c for k in ("error", "latency-90", "cpu", "mem", "workload"))}
        rec["all_error_metric_columns"] = [c for c in m.columns if "error" in c][:20]
        try:
            logs = pd.read_parquet(dl(f"{case}/logs.parquet"))
        except RemoteEntryNotFoundError:
            logs = None
        if logs is not None:
            lt = seconds(logs["timestamp"])
            lpre, lpost = (lt >= inject - W) & (lt < inject), (lt >= inject) & (lt < inject + W)
            msg = logs["message"].astype(str)
            err = msg.str.contains(ERR)
            svc = logs["container_name"].astype(str)
            is_rc = svc.str.contains(rc.replace("ts-", "").replace("service", "").strip("-"), case=False, regex=False)
            rec["log_err_all_pre_post"] = [int((lpre & err).sum()), int((lpost & err).sum())]
            rec["log_rc_lines_pre_post"] = [int((lpre & is_rc).sum()), int((lpost & is_rc).sum())]
            rec["log_err_rc_pre_post"] = [int((lpre & err & is_rc).sum()), int((lpost & err & is_rc).sum())]
            pre_t = set(msg[lpre].map(template))
            new = msg[lpost & ~msg.map(template).isin(pre_t)]
            rec["new_templates_after_injection"] = int(new.map(template).nunique())
            rec["examples_new_post_lines"] = [f"[{s}] {x[:200]}" for s, x in zip(svc[new.index][:6], new.head(6))]
        out.append(rec)
    (OUT / "probe_re2_signal.json").write_text(json.dumps(out, indent=1, default=str))
    for r in out:
        print(json.dumps(r, default=str)[:1500])


if __name__ == "__main__":
    main()
