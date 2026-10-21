"""Explanation study, amendment A2: rules translate, the LLM phrases (P3).

Every evidence item, the selected action and the decisive reason get a deterministic plain-language description
(`plain`), written by fixed rules from the record; values that AAF derived (relative-change or text facts) are
described as such, never as measurements. P3 must use those descriptions instead of translating on its own.
Arms: T (template), P1 (frozen prompt), P3. Same measures and judge as E1/A1.
"""
from __future__ import annotations

import copy
import json
import random
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402

from external_validation.explanations import run_explanations as R  # noqa: E402
from external_validation.explanations import run_explanations_v2 as V  # noqa: E402

HERE = Path(__file__).resolve().parent
SEED = 20261024

ACTION_PLAIN = {
    "Rollback to stable deployment": "go back to the last version of the software that worked",
    "Block release and fix pipeline": "stop the release until the automated release process is fixed",
    "Mitigate and monitor": "contain the problem now (for example restart, reroute or limit traffic) and keep watching it",
    "Scale adjustment": "add capacity (more servers or resources) for the affected service",
    "Review scaling policy": "review how the service adds and removes capacity, because costs are growing",
    "Patch or block release": "stop the release until the security or compliance problem is fixed",
    "No action (observe)": "make no change for now and keep watching",
    "Escalate for evidence/human review": "ask a person to review, because the evidence is not enough to decide",
}
RULE_PLAIN = {
    "release_security_override": "a release-check failure together with a security problem means the release must not go ahead",
    "deployment_reliability_causal_chain": "the service started crashing repeatedly right after a release while users were affected, which points to the release",
    "deployment_temporal_instability": "the service has been crashing and restarting repeatedly in a short time after a release",
    "reliability_resource_tradeoff": "the service is struggling and is also short of resources, so adding capacity addresses both",
    "multi_domain_accumulation": "several smaller warning signs across areas add up",
}


def plain_item(it: dict[str, Any]) -> str:
    f, v, src = it["field"], it["value"], str(it.get("source", ""))
    if src.startswith("relative:"):
        return {"p95_latency_ms": "responses are much slower than this system's normal level (a relative jump, not a measured time)",
                "saturation_pct": "CPU or memory use jumped far above this system's normal level (a relative jump, not a measured percentage)"}.get(f, f"{f} is far above normal")
    if src.startswith("categorical:"):
        return f"responders' notes report it ({src.split(':', 1)[1]}); the value is AAF's standard level for that fact, not a measurement"
    if isinstance(v, bool):
        yes = {"pipeline_failed": "the automated release checks failed", "config_drift": "the live configuration differs from the approved one",
               "artifact_mismatch": "the deployed package is not the approved one", "rollback_marker": "the release has been flagged for rollback",
               "policy_violation": "the release breaks a required security or compliance policy", "iam_drift": "access permissions changed from the approved setup",
               "compliance_gap": "a compliance requirement is not met"}
        no = {"pipeline_failed": "the automated release checks passed", "config_drift": "the configuration matches the approved one",
              "artifact_mismatch": "the deployed package is the approved one", "rollback_marker": "the release is not flagged for rollback",
              "policy_violation": "no security or compliance policy is broken", "iam_drift": "access permissions match the approved setup",
              "compliance_gap": "no compliance gap is reported"}
        return (yes if v else no).get(f, f"{f} is {v}")
    x = float(v)
    return {
        "p95_latency_ms": f"the slowest 5% of requests take about {x / 1000:.1f} seconds" if x >= 1000 else f"the slowest 5% of requests take about {x:.0f} milliseconds",
        "error_rate_pct": f"about {x:g}% of requests fail",
        "saturation_pct": f"the servers are running at about {x:g}% of their capacity",
        "availability_pct": f"the service was available {x:g}% of the time",
        "cost_spike_pct": f"costs are about {x:g}% above normal",
        "hpa_scale_to": f"the system automatically grew to {x:g} copies of the service",
        "cpu_request_increase_pct": f"the CPU the service asks for went up {x:g}%",
        "memory_request_increase_pct": f"the memory the service asks for went up {x:g}%",
        "critical_cves": f"{x:g} serious known security flaw(s) in the release",
        "restart_loops": f"the service crashed and restarted {x:g} times",
        "restart_burst_count": f"{x:g} restarts happened in a short burst",
        "restart_window_seconds": f"those restarts happened within {x:g} seconds",
    }.get(f, f"{f} = {v}")


def plain_reason(rec: dict[str, Any]) -> str:
    d = rec["decisive_reason"]
    if d.startswith("materiality gate"):
        return "no signal is serious enough to act on" if "no domain-specific" in d else "there is not enough usable evidence to decide"
    if d.startswith("cross-domain rule"):
        name = d.split(":")[0].replace("cross-domain rule ", "").strip()
        return RULE_PLAIN.get(name, "a rule that combines evidence from several areas decided it")
    dom = d.rsplit("dominant domain ", 1)[-1].split(" ")[0] if "dominant domain" in d else ""
    area = {"deployment": "the release", "reliability": "how well the service is working for users", "cost": "cost and resources", "security": "security"}.get(dom, "the evidence")
    return f"it is the best-supported option, and the main concern is {area}"


def augment(rec: dict[str, Any]) -> dict[str, Any]:
    rec = copy.deepcopy(rec)
    for it in rec["evidence"]:
        it["plain"] = plain_item(it)
    rec["plain_action"] = ACTION_PLAIN[rec["selected_action"]]
    rec["plain_reason"] = plain_reason(rec)
    rec["plain_alternatives"] = {c["action"]: ACTION_PLAIN.get(c["action"], c["action"]) for c in rec["decision_path"]["top_candidates"] if c["action"] != rec["selected_action"]}
    return rec


ANSWER_SYSTEM_P3 = V.ANSWER_SYSTEM_P2.replace(
    "- Translate technical terms instead of repeating them, e.g. \"p95 latency 980 ms\" -> \"the slowest requests take about a\n  second\"; \"error rate 16%\" -> \"about 1 in 6 requests fail\"; \"saturation 93%\" -> \"the servers are almost at full\n  capacity\"; \"critical CVE\" -> \"a serious known security flaw\"; \"pipeline failed\" -> \"the automated release checks failed\".",
    "- Use the plain-language descriptions supplied in the record (evidence[].plain, plain_action, plain_reason,\n  plain_alternatives) instead of translating anything yourself. Do not reinterpret numbers; repeat them only as written\n  in a plain description. Name the action exactly and then say what it means using plain_action.")
assert ANSWER_SYSTEM_P3 != V.ANSWER_SYSTEM_P2


def records() -> list[dict[str, Any]]:
    recs = []
    for f in ("cases_v1.yaml", "cases_replication_v1.yaml"):
        for c in yaml.safe_load((ROOT / "agentic_experiments" / f).read_text())["cases"]:
            recs.append(V.full_record(c["id"], "structured", c["stratum"], copy.deepcopy(c["evidence"]), None, True))
    import pandas as pd
    from huggingface_hub import hf_hub_download
    from external_validation.aaf_v3.rcaeval_v3 import v3_telemetry
    from external_validation.aaf_v3.run_rcaeval_v3 import normalise_columns
    from external_validation.rcaeval.adapter import adapt
    used = set()
    for p in (HERE / "results").glob("*/records.json"):
        used |= {r["record_id"] for r in json.loads(p.read_text())}
    w = pd.read_csv(next((ROOT / "external_validation/aaf_v3/results/rcaeval").glob("RE1_RE3_*/window_results.csv")))
    rng = random.Random(SEED)
    for suite in ("RE1", "RE3"):
        for win in ("fault", "control"):
            rows = [r for r in w[(w.suite == suite) & (w.window == win)].sort_values("case").to_dict("records") if f"{r['case']}:{r['window']}" not in used]
            for r in rng.sample(rows, 6):
                dl = lambda p: Path(hf_hub_download("phamquiluan/RCAEval", p, repo_type="dataset"))
                inject = int(float(dl(f"{r['case']}/inject_time.txt").read_text().strip()))
                m, _ = normalise_columns(pd.read_parquet(dl(f"{r['case']}/metrics.parquet")))
                tel, diag = adapt(m, inject if win == "fault" else inject - 300, logs=None)
                tel3, _ = v3_telemetry(tel, diag)
                rec = V.full_record(f"{r['case']}:{win}", "rcaeval", f"{suite}-{win}", tel3, R.WEIGHTS_RCA, False)
                assert rec["selected_action"] == r["v3_action"]
                recs.append(rec)
    return [augment(r) for r in recs]


if __name__ == "__main__":
    import os
    os.environ.setdefault("EXPL_PREFIX", "plain_rules_")
    os.environ.setdefault("EXPL_AMENDMENT", "amendment A2")
    os.environ.setdefault("ARM2_NAME", "P3")
    V.fresh_records = records
    V.ANSWER_SYSTEM_P2 = ANSWER_SYSTEM_P3  # arm labelled "P2" in the output files is P3 (see metadata prompt hash)
    V.HERE = HERE
    V.main()
