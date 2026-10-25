"""Study 9 deciders. Reads only incident snapshots (replicate 1); never reads outcomes. Writes one decision row per
(decider, case variant, run). LLM arms use Qwen via Ollama (model_replication.run_qwen.QwenRuntime).
Sharded by SHARD / NSHARDS over (variant, LLM arm, run) jobs; deterministic deciders are computed in shard 0."""
from __future__ import annotations

import copy
import json
import os
import random
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agentic_experiments import agentic_runtime as AR  # noqa: E402
from agentic_experiments.hybrid import deterministic_assessment, missing_fields, run_agentic_only, run_hybrid, selective_trigger  # noqa: E402
from orchestrator.decision_baselines import choose_dominant_domain_action  # noqa: E402
from revision_2026 import run_llm_arms as RA  # noqa: E402
from revision_2026.autofetch_baseline import FIELD_TO_TOOL  # noqa: E402
from agentic_experiments.rules_v12 import deterministic_assessment_v12, run_hybrid_v12  # noqa: E402
from agentic_experiments.rules_v14 import deterministic_assessment_v14, run_hybrid_v14  # noqa: E402

HERE = ROOT / os.environ.get("OUTCOME_DIR", "outcome_study")
TOOLS = sorted(set().union(*AR.DOMAIN_TOOLS.values()))
F2T = dict(FIELD_TO_TOOL, **{"deploy.restart_loops": "get_restart_history"})  # same tool family; field not in the frozen map
BLOCK_OF = {"D-DevOps": "deploy", "D-SRE": "sre", "D-FinOps": "finops", "D-Sec": "sec"}
WITHHOLD = [  # fixed order; the first applicable is withheld (PROTOCOL.md, Study 9b)
    ("vuln", ["sec.critical_cves"], "get_vulnerability_details"),
    ("secret", ["sec.policy_violation", "deploy.config_drift"], "get_policy_gate_status"),
    ("crash", ["deploy.restart_loops"], "get_restart_history"),
    ("reliability", ["sre.p95_latency_ms", "sre.saturation_pct"], "get_latency_history"),
]


def m(status: str, src: str) -> dict:
    return {"status": status, "source": src}


def evidence(s: dict) -> dict:
    h = s["http"]
    sre = {"p95_latency_ms": h["latency_p95_ms"], "error_rate_pct": h["error_rate_pct"], "availability_pct": h["availability_pct"],
           "_evidence": {k: m("measured", "HTTP load measurement") for k in ("p95_latency_ms", "error_rate_pct", "availability_pct")}}
    if s.get("pre_release"):  # Study 14: measured pre-release baseline of the same service
        sre["p95_pre_release_ms"] = s["pre_release"]["p95_ms"]
        sre["error_rate_pre_release_pct"] = s["pre_release"]["error_rate_pct"]
        for k in ("p95_pre_release_ms", "error_rate_pre_release_pct"):
            sre["_evidence"][k] = m("measured", "HTTP load measurement before the release")
    if s.get("saturation_pct") is not None:
        sre["saturation_pct"] = s["saturation_pct"]
        sre["_evidence"]["saturation_pct"] = m("measured", "metrics-server CPU / limit")
    deploy = {"restart_loops": s["restarts"], "config_drift": bool(s["secret_in_config"]),
              "_evidence": {"restart_loops": m("measured", "container restart count"), "config_drift": m("measured", "deployment environment")}}
    if "recent_release" in s:  # measured from ReplicaSet history (Study 12 onward)
        deploy["recent_release"] = bool(s["recent_release"])
        deploy["_evidence"]["recent_release"] = m("measured", "ReplicaSet history: newest vs previous pod spec")
    cost = 100.0 * (s["footprint_cpu_m"] / 100.0 - 1.0)
    finops = {"cost_spike_pct": cost, "cpu_request_increase_pct": cost,
              "_evidence": {"cost_spike_pct": m("proxy", "replicas x CPU request vs healthy"), "cpu_request_increase_pct": m("proxy", "replicas x CPU request vs healthy")}}
    sec = {"critical_cves": s["critical_cves_new"], "policy_violation": bool(s["secret_in_config"]),
           "_evidence": {"critical_cves": m("measured", "trivy image scan"), "policy_violation": m("measured", "configuration policy check")}}
    if s.get("noncritical_cves") is not None:  # Study 13: HIGH+MEDIUM findings (AAF's rules do not treat them as material)
        sec["noncritical_cves"] = s["noncritical_cves"]
        sec["_evidence"]["noncritical_cves"] = m("measured", "trivy image scan (HIGH, MEDIUM)")
    return {"deploy": deploy, "sre": sre, "finops": finops, "sec": sec}


def fault_kinds(case: dict, snap: dict) -> list[str]:
    cur = set(case["current"])
    kinds = [k for k in ("vuln", "secret", "crash") if k in cur]
    if "perf" in cur or case["load"] == "high":
        kinds.append("reliability")
    return kinds


def variants(case: dict, snap: dict) -> list[dict]:
    ev = evidence(snap)
    out = [{"id": case["id"], "variant": "full", "evidence": ev, "available_tools": TOOLS}]
    kinds = fault_kinds(case, snap)
    for kind, fields, tool in WITHHOLD:
        if kind in kinds:
            ev2 = copy.deepcopy(ev)
            result = {"id": f"TOOL-{case['id']}-{tool}"}
            for f in fields:
                block, field = f.split(".")
                if field in ev2[block]:
                    result.setdefault(block, {})[field] = ev2[block][field]
                    ev2[block][field] = None
                    ev2[block]["_evidence"][field] = {"status": "missing"}
            out.append({"id": case["id"] + "-W", "variant": "withheld", "evidence": ev2, "available_tools": TOOLS,
                        "expected_tool": tool, "tool_result": result, "withheld": fields})
            break
    for v in out:
        v["stratum"] = "OUTCOME"
    return out


def b2(v: dict) -> str:
    ev = copy.deepcopy(v["evidence"])
    called = set()
    for f in missing_fields(ev):
        tool = F2T.get(f)
        if tool and tool not in called:
            called.add(tool)
            if tool == v.get("expected_tool"):
                ev = AR.merge_tool_result(ev, copy.deepcopy(v["tool_result"]))
    return str(deterministic_assessment(ev)["selected_action"])


def deterministic_rows(v: dict) -> list[dict]:
    ev = v["evidence"]
    rows = {"B1": deterministic_assessment(copy.deepcopy(ev))["selected_action"], "B2": b2(v),
            "B1-12": deterministic_assessment_v12(copy.deepcopy(ev))["selected_action"],
            "B1-14": deterministic_assessment_v14(copy.deepcopy(ev))["selected_action"],
            "Dominant": choose_dominant_domain_action(copy.deepcopy(ev))[0],
            "AlwaysRollback": "Rollback to stable deployment", "AlwaysNone": "No action (observe)"}
    if "noncritical_cves" in ev["sec"]:  # Study 13 fixed policy: block on any vulnerability finding
        anysec = (ev["sec"]["critical_cves"] or 0) > 0 or (ev["sec"]["noncritical_cves"] or 0) > 0 or bool(ev["sec"]["policy_violation"])
        rows["BlockAnyCVE"] = "Patch or block release" if anysec else "No action (observe)"
    for name, block in BLOCK_OF.items():
        rows[name] = deterministic_assessment({block: copy.deepcopy(ev[block])})["selected_action"]
    return [{"decider": k, "case": v["id"], "variant": v["variant"], "run": 0, "action": str(a), "error": None} for k, a in rows.items()]


def main() -> None:
    from model_replication.run_qwen import QwenRuntime
    snaps = sorted((HERE / "measurements" / "rep1").glob("*.json"))
    cfg = {c["id"]: c for c in yaml.safe_load((HERE / "cases.yaml").read_text())["cases"]}
    allv = []
    for p in snaps:
        d = json.loads(p.read_text())
        allv += variants(cfg[d["case"]["id"]], d["snapshot"])
    shard, nshards = int(os.environ.get("SHARD", "0")), int(os.environ.get("NSHARDS", "1"))
    out = HERE / os.environ.get("DECISIONS_DIR", "decisions")
    out.mkdir(parents=True, exist_ok=True)
    (out / "variants.json").write_text(json.dumps(allv, indent=1, default=str)) if shard == 0 else None
    if shard == 0:
        rows = [r for v in allv for r in deterministic_rows(v)]
        (out / "deterministic.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    rt = {"AAF": QwenRuntime(False), "AAF-12": QwenRuntime(False), "AAF-14": QwenRuntime(False), "B5": QwenRuntime(False), "B6": QwenRuntime(True, RA.COORD_B6)}
    fn = {"AAF": lambda c: run_hybrid(c, rt["AAF"]), "AAF-12": lambda c: run_hybrid_v12(c, rt["AAF-12"]),
          "AAF-14": lambda c: run_hybrid_v14(c, rt["AAF-14"]),
          "B5": lambda c: run_agentic_only(c, rt["B5"]), "B6": lambda c: run_agentic_only(c, rt["B6"])}
    arms = os.environ.get("LLM_ARMS", "AAF B5 B6").split()
    jobs = [(arm, v, run) for v in allv for arm in arms for run in (1, 2, 3)]
    random.Random(9).shuffle(jobs)
    path = out / f"llm_shard_{shard:02d}.jsonl"
    done = set()
    if path.exists():
        done = {(r["decider"], r["case"], r["run"]) for r in map(json.loads, path.read_text().splitlines()) if r.get("error") is None}
    with path.open("a") as fh:
        for arm, v, run in jobs[shard::nshards]:
            if (arm, v["id"], run) in done:
                continue
            try:
                res = fn[arm](v)
                row = {"decider": arm, "case": v["id"], "variant": v["variant"], "run": run, "action": res["selected_action"], **RA.slim(res),
                       "trigger": (res.get("trigger") or {}).get("reasons"), "error": None}
                if arm == "B5":  # B3: deterministic re-decision on B5's evidence (with any tool result B5's agents fetched)
                    ev = copy.deepcopy(v["evidence"])
                    for t in res.get("tool_events") or []:
                        if t.get("tool") == v.get("expected_tool") and isinstance(t.get("result"), dict):
                            ev = AR.merge_tool_result(ev, t["result"])
                    row["b3_action"] = str(deterministic_assessment(ev)["selected_action"])
            except Exception as exc:
                row = {"decider": arm, "case": v["id"], "variant": v["variant"], "run": run, "action": None, "error": f"{type(exc).__name__}: {str(exc)[:300]}"}
            fh.write(json.dumps(row, default=str) + "\n"); fh.flush()
            print(arm, v["id"], run, row["action"], flush=True)


if __name__ == "__main__":
    main()
