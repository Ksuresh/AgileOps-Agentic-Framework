"""Study 8 fresh test cases, generated from a fixed template grid with a fixed seed (no hand-picking of cases).

Unflagged-gap templates (decisive evidence hidden from the decision-time view, no 'missing' flag):
  T1 deployment cause of degradation  visible: reliability degraded        hidden: deploy (restarts after deploy / drift)  -> Rollback
  T2 hidden security block            visible: pipeline failed or healthy   hidden: critical CVEs + policy violation        -> Patch or block release
  T3 hidden reliability under cost    visible: moderate cost growth         hidden: degraded reliability + saturation       -> Scale adjustment
  T4 hidden cost under degradation    visible: degraded reliability         hidden: large cost growth + replicas            -> Scale adjustment
  T8 drift with hidden degradation    visible: configuration drift          hidden: degraded reliability                    -> Rollback
  T5 resolved reliability alarm       visible: stale bad reliability        fresh: healthy                                  -> No action (observe)
  T6 resolved policy violation        visible: stale policy violation       fresh: resolved                                 -> No action (observe)
  T7 resolved cost spike              visible: stale cost spike             fresh: normal                                   -> No action (observe)
Gap types: absent (the hidden domain block is not in the evidence), cross (the hidden domain block is present but
lacks the decisive fields), stale (the hidden fields are present with status 'stale' and benign values).
Grid: T1,T2,T3,T4,T8 x {absent, cross, stale} = 15; T5,T6,T7 x {stale} = 3; plus 2 drawn at random from the 15
combinations (new random values) = 20 decisive cases. Controls (fetch confirms the visible action): C1-C4.
Standard strata: 3 CLEAR, 3 AMBIGUOUS, 3 INCOMPLETE (flagged missing), 3 MISLEADING, values jittered by seed.
Each case lists 3-4 approved tools: the evidence-bearing tool plus seeded distractors.
"""
import random
from pathlib import Path

import yaml

R, B, M = "Rollback to stable deployment", "Block release and fix pipeline", "Mitigate and monitor"
S, V, P, N = "Scale adjustment", "Review scaling policy", "Patch or block release", "No action (observe)"
TOOLS = ["get_restart_history", "get_pipeline_history", "get_deployment_history", "get_error_rate_history", "get_latency_history",
         "get_saturation_history", "get_replica_history", "get_cost_proxy_history", "get_policy_gate_status", "get_vulnerability_details"]
rng = random.Random(20261023)
J = lambda lo, hi: rng.randint(lo, hi)
F = lambda lo, hi, d=1: round(rng.uniform(lo, hi), d)


def tools_for(expected, k=None):
    k = k or rng.choice([3, 4])
    others = rng.sample([t for t in TOOLS if t != expected], k - 1)
    out = others + [expected]
    rng.shuffle(out)
    return out


def st(age):
    return {"status": "stale", "age_minutes": age}


def sre_bad():
    return {"p95_latency_ms": J(820, 1050), "error_rate_pct": F(10.5, 15), "saturation_pct": J(70, 86), "availability_pct": F(97.0, 98.7)}


def sre_bad_sat():
    return {"p95_latency_ms": J(800, 980), "error_rate_pct": F(9, 13), "saturation_pct": J(92, 97)}


def sre_ok():
    return {"p95_latency_ms": J(190, 260), "error_rate_pct": F(0.4, 1.5), "saturation_pct": J(35, 55), "availability_pct": F(99.85, 99.99, 2)}


def case(cid, stratum, sub, evidence, author, expected=None, result=None, template=None):
    c = {"id": cid, "stratum": stratum, "substratum": sub, "template": template, "evidence": evidence}
    if expected:
        c.update({"available_tools": tools_for(expected), "expected_tool": expected, "tool_result": {"id": f"TOOL-{cid}", **result}})
    return c, author


def unflagged(t, gap, cid):
    if t == "T1":
        hidden = rng.choice(["restarts", "drift"])
        if hidden == "restarts":
            tool, res = "get_restart_history", {"deploy": {"restart_burst_count": J(5, 7), "restart_window_seconds": J(25, 45), "deployment_preceded_restarts": True}}
        else:
            tool, res = "get_deployment_history", {"deploy": {"config_drift": True, "deployment_preceded_restarts": True, "restart_burst_count": J(4, 6), "restart_window_seconds": J(30, 50)}}
        ev = {"sre": sre_bad()}
        if gap == "cross":
            ev["deploy"] = {"pipeline_failed": False}
        elif gap == "stale":
            ev["deploy"] = {"pipeline_failed": False, "config_drift": False, "restart_burst_count": 0, "restart_window_seconds": 300,
                            "_evidence": {"config_drift": st(J(1440, 4320)), "restart_burst_count": st(J(1440, 4320)), "restart_window_seconds": st(J(1440, 4320))}}
            tool, res = "get_deployment_history", {"deploy": {"config_drift": True, "deployment_preceded_restarts": True, "restart_burst_count": J(4, 6), "restart_window_seconds": J(30, 50)}}
        return case(cid, "UNMARKED", gap, ev, [R], tool, res, t)
    if t == "T2":
        tool = rng.choice(["get_vulnerability_details", "get_policy_gate_status"])
        res = {"sec": {"critical_cves": J(1, 3), "policy_violation": True}}
        ev = {"deploy": {"pipeline_failed": rng.choice([True, False]), "artifact_mismatch": False}, "sre": sre_ok()}
        if gap == "cross":
            ev["sec"] = {"iam_drift": False}
        elif gap == "stale":
            ev["sec"] = {"critical_cves": 0, "policy_violation": False, "_evidence": {"critical_cves": st(J(4320, 14400)), "policy_violation": st(J(4320, 14400))}}
        return case(cid, "UNMARKED", gap, ev, [P], tool, res, t)
    if t == "T3":
        tool = rng.choice(["get_saturation_history", "get_latency_history"])
        res = {"sre": sre_bad_sat()}
        ev = {"finops": {"cost_spike_pct": J(18, 26), "hpa_scale_to": J(9, 12)}, "deploy": {"pipeline_failed": False}}
        if gap == "cross":
            ev["sre"] = {"availability_pct": F(99.2, 99.6, 2)}
        elif gap == "stale":
            ok = sre_ok()
            ev["sre"] = {**ok, "_evidence": {k: st(J(180, 720)) for k in ok}}
        return case(cid, "UNMARKED", gap, ev, [S], tool, res, t)
    if t == "T4":
        tool = rng.choice(["get_cost_proxy_history", "get_replica_history"])
        res = {"finops": {"cost_spike_pct": J(30, 42), "hpa_scale_to": J(12, 16)}}
        ev = {"sre": {"p95_latency_ms": J(800, 950), "error_rate_pct": F(9.5, 12), "saturation_pct": J(88, 94)}, "deploy": {"pipeline_failed": False}}
        if gap == "cross":
            ev["finops"] = {"cpu_request_increase_pct": J(5, 15)}
        elif gap == "stale":
            ev["finops"] = {"cost_spike_pct": J(2, 6), "hpa_scale_to": J(4, 8), "_evidence": {"cost_spike_pct": st(J(1440, 4320)), "hpa_scale_to": st(J(1440, 4320))}}
        return case(cid, "UNMARKED", gap, ev, [S], tool, res, t)
    if t == "T8":
        tool = rng.choice(["get_error_rate_history", "get_latency_history"])
        res = {"sre": {"p95_latency_ms": J(820, 980), "error_rate_pct": F(11, 14), "availability_pct": F(97.5, 98.6)}}
        ev = {"deploy": {"config_drift": True, "pipeline_failed": False, "artifact_mismatch": False}, "finops": {"cost_spike_pct": J(1, 4)}}
        if gap == "cross":
            ev["sre"] = {"saturation_pct": J(40, 60)}
        elif gap == "stale":
            ok = sre_ok()
            ev["sre"] = {**ok, "_evidence": {k: st(J(120, 600)) for k in ok}}
        return case(cid, "UNMARKED", gap, ev, [R], tool, res, t)
    if t == "T5":
        tool = rng.choice(["get_latency_history", "get_error_rate_history"])
        bad = sre_bad()
        return case(cid, "UNMARKED", "stale", {"sre": {**bad, "_evidence": {k: st(J(300, 900)) for k in bad}}, "deploy": {"pipeline_failed": False, "config_drift": False}},
                    [N], tool, {"sre": sre_ok()}, t)
    if t == "T6":
        return case(cid, "UNMARKED", "stale", {"sec": {"critical_cves": 0, "policy_violation": True, "_evidence": {"policy_violation": st(J(7200, 20160))}},
                                               "deploy": {"pipeline_failed": False}, "sre": sre_ok()},
                    [N], "get_policy_gate_status", {"sec": {"policy_violation": False}}, t)
    if t == "T7":
        tool = rng.choice(["get_cost_proxy_history", "get_replica_history"])
        return case(cid, "UNMARKED", "stale", {"finops": {"cost_spike_pct": J(35, 45), "hpa_scale_to": J(14, 17), "_evidence": {"cost_spike_pct": st(J(2880, 7200)), "hpa_scale_to": st(J(2880, 7200))}},
                                               "sre": sre_ok(), "deploy": {"pipeline_failed": False}},
                    [N], tool, {"finops": {"cost_spike_pct": J(1, 4), "hpa_scale_to": J(3, 5)}}, t)
    raise ValueError(t)


C = []
grid = [(t, g) for t in ("T1", "T2", "T3", "T4", "T8") for g in ("absent", "cross", "stale")] + [("T5", "stale"), ("T6", "stale"), ("T7", "stale")]
grid += rng.sample(grid[:15], 2)
for i, (t, g) in enumerate(grid, 1):
    C.append(unflagged(t, g, f"FX-U{i:02d}"))
# controls
C.append(case("FX-K01", "UNMARKED", "control", {"sre": sre_bad(), "deploy": {"pipeline_failed": False}}, [M], "get_deployment_history",
              {"deploy": {"config_drift": False, "deployment_preceded_restarts": False}}, "C1"))
C.append(case("FX-K02", "UNMARKED", "control", {"deploy": {"pipeline_failed": True, "artifact_mismatch": False}, "sre": sre_ok()}, [B], "get_vulnerability_details",
              {"sec": {"critical_cves": 0, "policy_violation": False}}, "C2"))
ok = sre_ok()
C.append(case("FX-K03", "UNMARKED", "control", {"sre": {**ok, "_evidence": {k: st(J(20, 50)) for k in ok}}, "deploy": {"pipeline_failed": False}}, [N], "get_latency_history",
              {"sre": sre_ok()}, "C3"))
C.append(case("FX-K04", "UNMARKED", "control", {"sec": {"critical_cves": J(1, 3), "policy_violation": True}, "sre": sre_ok()}, [P], "get_pipeline_history",
              {"deploy": {"pipeline_failed": True}}, "C4"))
# standard strata
C.append(case("FX-C01", "CLEAR", "standard", {"deploy": {"pipeline_failed": False, "config_drift": False}, "sre": sre_ok(), "finops": {"cost_spike_pct": J(1, 4)}, "sec": {"critical_cves": 0, "policy_violation": False}}, [N]))
C.append(case("FX-C02", "CLEAR", "standard", {"sre": {"p95_latency_ms": J(1000, 1200), "error_rate_pct": F(13, 17), "saturation_pct": J(90, 96), "availability_pct": F(95.5, 97.5)}, "deploy": {"pipeline_failed": False}, "sec": {"policy_violation": False}}, [M]))
C.append(case("FX-C03", "CLEAR", "standard", {"sec": {"critical_cves": J(2, 4), "policy_violation": True}, "deploy": {"pipeline_failed": False}, "sre": sre_ok()}, [P]))
C.append(case("FX-A01", "AMBIGUOUS", "standard", {"sre": {"p95_latency_ms": J(455, 475), "error_rate_pct": F(7.6, 8.2), "saturation_pct": J(83, 87), "availability_pct": F(99.0, 99.15, 2)}, "deploy": {"pipeline_failed": False}}, [M]))
C.append(case("FX-A02", "AMBIGUOUS", "standard", {"finops": {"cost_spike_pct": J(20, 24), "hpa_scale_to": J(10, 12), "cpu_request_increase_pct": J(46, 52)}, "sre": sre_ok()}, [V, S]))
C.append(case("FX-A03", "AMBIGUOUS", "standard", {"deploy": {"config_drift": True, "pipeline_failed": False}, "sre": {"p95_latency_ms": J(450, 470), "error_rate_pct": F(7.8, 8.3), "availability_pct": F(99.1, 99.4, 2)}}, [R, M]))
C.append(case("FX-I01", "INCOMPLETE", "standard", {"sre": {"p95_latency_ms": J(850, 950), "error_rate_pct": None, "saturation_pct": J(86, 92), "_evidence": {"error_rate_pct": {"status": "missing"}}}},
              [M], "get_error_rate_history", {"sre": {"error_rate_pct": F(11, 14)}}))
C.append(case("FX-I02", "INCOMPLETE", "standard", {"finops": {"cost_spike_pct": None, "hpa_scale_to": J(14, 16), "cpu_request_increase_pct": J(55, 62), "_evidence": {"cost_spike_pct": {"status": "missing"}}}, "sre": sre_ok()},
              [S, V], "get_cost_proxy_history", {"finops": {"cost_spike_pct": J(36, 44)}}))
C.append(case("FX-I03", "INCOMPLETE", "standard", {"sec": {"critical_cves": None, "policy_violation": True, "_evidence": {"critical_cves": {"status": "missing"}}}, "deploy": {"pipeline_failed": True}},
              [P], "get_vulnerability_details", {"sec": {"critical_cves": J(1, 3)}}))
C.append(case("FX-M01", "MISLEADING", "standard", {"finops": {"cost_spike_pct": J(36, 44), "hpa_scale_to": J(14, 16)}, "sre": {"p95_latency_ms": J(920, 1000), "error_rate_pct": F(12.5, 14.5), "saturation_pct": J(93, 96)}, "sec": {"policy_violation": False}}, [S]))
C.append(case("FX-M02", "MISLEADING", "standard", {"finops": {"cost_spike_pct": J(1, 4), "hpa_scale_to": J(12, 14), "cpu_request_increase_pct": J(2, 6)}, "sre": sre_ok(), "deploy": {"pipeline_failed": False}}, [N, V]))
C.append(case("FX-M03", "MISLEADING", "standard", {"deploy": {"restart_burst_count": 2, "restart_window_seconds": J(95, 130), "deployment_preceded_restarts": False}, "sre": {"p95_latency_ms": J(850, 920), "error_rate_pct": F(10.5, 12), "saturation_pct": J(86, 90)}}, [M]))

cases = [c for c, _ in C]
assert len(cases) == 36 and len({c["id"] for c in cases}) == 36
out = Path(__file__).parent
(out / "cases_fx_v1.yaml").write_text("# FROZEN BEFORE ANY RUN. Generated by make_cases.py (seed 20261023). See PROTOCOL.md.\n" + yaml.safe_dump({"version": 1, "n_cases": 36, "cases": cases}, sort_keys=False, width=220))
(out / "author_labels.yaml").write_text("# Author admissible actions, sealed with the cases before any run (primary key).\n" + yaml.safe_dump({c["id"]: a for c, a in C}, sort_keys=True, width=220))
print("wrote", len(cases))
