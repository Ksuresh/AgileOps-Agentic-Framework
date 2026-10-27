"""Builds the blind labelling page (label_page.html) and the code->case map (label_map.json).
The page shows only opaque codes and the evidence; no ids, strata, focus, oracle, or AAF output."""
import json
import random
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from agentic_experiments.agentic_runtime import ACTIONS  # noqa: E402

FIELDS = {
    "deploy": ("Delivery (DevOps)", {
        "pipeline_failed": ("CI/CD pipeline failed", None), "artifact_mismatch": ("Deployed artifact differs from the approved release", None),
        "config_drift": ("Configuration drift from the approved baseline", None), "restart_burst_count": ("Pod restarts in a burst", "restarts"),
        "restart_window_seconds": ("Time window of those restarts", "s"), "deployment_preceded_restarts": ("A deployment happened just before the restarts", None),
        "rollback_marker": ("Rollback marker present", None)}),
    "sre": ("Reliability (SRE)", {
        "p95_latency_ms": ("p95 latency", "ms"), "error_rate_pct": ("Error rate", "%"), "saturation_pct": ("Resource saturation (CPU/memory)", "%"),
        "availability_pct": ("Availability", "%")}),
    "finops": ("Cost and capacity (FinOps)", {
        "cost_spike_pct": ("Cost increase vs. normal", "%"), "hpa_scale_to": ("Autoscaler scaled to", "replicas"),
        "cpu_request_increase_pct": ("CPU requests increase", "%"), "memory_request_increase_pct": ("Memory requests increase", "%")}),
    "sec": ("Security (DevSecOps)", {
        "critical_cves": ("Critical CVEs", None), "policy_violation": ("Security policy violation", None), "iam_drift": ("IAM permission drift", None),
        "compliance_gap": ("Compliance gap", None)}),
}


def fmt(v, unit):
    if isinstance(v, bool):
        return "Yes" if v else "No"
    if v is None:
        return None
    return f"{v} {unit}".strip() if unit else str(v)


def age(m):
    return f"{m // 1440} days" if m >= 1440 else (f"{m // 60} hours" if m >= 60 else f"{m} minutes")


def render(evidence, result):
    out = []
    for dom, (title, spec) in FIELDS.items():
        block = evidence.get(dom) or {}
        res = (result or {}).get(dom) or {}
        meta = block.get("_evidence", {}) or {}
        rows = []
        for f, (label, unit) in spec.items():
            if f not in block and f not in res:
                continue
            st = (meta.get(f) or {}).get("status")
            now = fmt(block.get(f), unit) if f in block else None
            note = None
            if f in block and st == "missing":
                now, note = None, "missing"
            elif f in block and st == "stale":
                note = f"last updated {age(int(meta[f].get('age_minutes', 0)))} ago"
            elif f not in block:
                note = "not in the current evidence"
            rows.append({"label": label, "now": now, "note": note, "check": fmt(res[f], unit) if f in res else None})
        if rows or dom in evidence:
            out.append({"domain": title, "rows": rows, "absent": dom not in evidence})
    return out


def load(path, src):
    return [(src, c) for c in yaml.safe_load((ROOT / path).read_text())["cases"]]


items = load("agentic_experiments/cases_v1.yaml", "orig") + load("agentic_experiments/cases_replication_v1.yaml", "orig") + \
    load("external_validation/expert_study/cases_new_v1.yaml", "new")
rng = random.Random(20261022)
rng.shuffle(items)
page, mapping = [], {}
for k, (src, c) in enumerate(items, 1):
    code = f"K{k:02d}"
    mapping[code] = {"case": c["id"], "source": src}
    page.append({"code": code, "domains": render(c["evidence"], c.get("tool_result"))})
(HERE / "label_map.json").write_text(json.dumps(mapping, indent=1))
data = json.dumps({"actions": ACTIONS, "cases": page}, separators=(",", ":"))
tpl = (HERE / "label_page_template.html").read_text()
(HERE / "label_page.html").write_text(tpl.replace("/*__DATA__*/null", data))
print(len(page), "cases;", len(data), "bytes of data")
