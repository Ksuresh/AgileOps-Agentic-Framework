"""Writes cases_new_v1.yaml (48 new cases) and author_labels.yaml. Run once; outputs are frozen by commit.
Author-proposed admissible actions are kept in author_labels.yaml only (sealed); the primary answer key for the new
cases is the external expert's blind labels. Evaluator-only fields (expected_tool, tool_result) follow the original
case format. Stale evidence is marked with status 'stale' and age_minutes (not 'missing')."""
import yaml
from pathlib import Path

R = "Rollback to stable deployment"; B = "Block release and fix pipeline"; M = "Mitigate and monitor"
S = "Scale adjustment"; V = "Review scaling policy"; P = "Patch or block release"; N = "No action (observe)"
T_ALL = None


def st(age):  # stale marker
    return {"status": "stale", "age_minutes": age}


C = []


def case(i, stratum, sub, evidence, author, tools=None, exp=None, result=None):
    c = {"id": i, "stratum": stratum, "substratum": sub, "evidence": evidence}
    if tools:
        c.update({"available_tools": tools, "expected_tool": exp, "tool_result": dict({"id": f"TOOL-{i}"}, **result)})
    C.append((c, author))


ok_dep = {"pipeline_failed": False, "artifact_mismatch": False, "config_drift": False}
ok_sre = {"p95_latency_ms": 215, "error_rate_pct": 0.7, "saturation_pct": 41, "availability_pct": 99.95}
ok_fin = {"cost_spike_pct": 2, "hpa_scale_to": 4}
ok_sec = {"critical_cves": 0, "policy_violation": False}

# ---------------- STANDARD: CLEAR ----------------
case("NV-C01", "CLEAR", "standard", {"deploy": ok_dep, "sre": ok_sre, "finops": ok_fin, "sec": ok_sec}, [N])
case("NV-C02", "CLEAR", "standard", {"sre": {"p95_latency_ms": 1100, "error_rate_pct": 14, "saturation_pct": 91, "availability_pct": 96.5}, "deploy": {"pipeline_failed": False}, "finops": {"cost_spike_pct": 3}, "sec": ok_sec}, [M])
case("NV-C03", "CLEAR", "standard", {"sec": {"critical_cves": 3, "policy_violation": True, "iam_drift": False}, "deploy": ok_dep, "sre": ok_sre, "finops": ok_fin}, [P])
case("NV-C04", "CLEAR", "standard", {"deploy": {"pipeline_failed": True, "artifact_mismatch": True, "config_drift": False}, "sre": ok_sre, "finops": ok_fin, "sec": ok_sec}, [B])
case("NV-C05", "CLEAR", "standard", {"deploy": {"restart_burst_count": 6, "restart_window_seconds": 40, "deployment_preceded_restarts": True, "pipeline_failed": False}, "sre": {"p95_latency_ms": 900, "error_rate_pct": 12, "saturation_pct": 80, "availability_pct": 98.0}, "finops": ok_fin, "sec": ok_sec}, [R])
case("NV-C06", "CLEAR", "standard", {"finops": {"cost_spike_pct": 42, "hpa_scale_to": 16, "cpu_request_increase_pct": 60}, "sre": ok_sre, "deploy": {"pipeline_failed": False}, "sec": ok_sec}, [S, V])
# ---------------- STANDARD: AMBIGUOUS ----------------
case("NV-A01", "AMBIGUOUS", "standard", {"sre": {"p95_latency_ms": 460, "error_rate_pct": 7.8, "saturation_pct": 85, "availability_pct": 99.05}, "deploy": {"pipeline_failed": False}, "finops": {"cost_spike_pct": 3}, "sec": ok_sec}, [M])
case("NV-A02", "AMBIGUOUS", "standard", {"finops": {"cost_spike_pct": 21, "hpa_scale_to": 11, "cpu_request_increase_pct": 48}, "sre": ok_sre, "deploy": {"pipeline_failed": False}, "sec": ok_sec}, [V, S])
case("NV-A03", "AMBIGUOUS", "standard", {"deploy": {"restart_burst_count": 3, "restart_window_seconds": 70, "pipeline_failed": False}, "sre": {"p95_latency_ms": 480, "error_rate_pct": 8.2, "saturation_pct": 78, "availability_pct": 99.2}, "finops": ok_fin, "sec": ok_sec}, [M, R])
case("NV-A04", "AMBIGUOUS", "standard", {"sre": {"p95_latency_ms": 820, "error_rate_pct": 9, "saturation_pct": 88, "availability_pct": 98.9}, "finops": {"cost_spike_pct": 24, "hpa_scale_to": 12}, "deploy": {"pipeline_failed": False}, "sec": ok_sec}, [S])
case("NV-A05", "AMBIGUOUS", "standard", {"sec": {"critical_cves": 1, "policy_violation": False, "compliance_gap": True}, "deploy": ok_dep, "sre": ok_sre, "finops": ok_fin}, [P])
case("NV-A06", "AMBIGUOUS", "standard", {"deploy": {"config_drift": True, "pipeline_failed": False}, "sre": {"p95_latency_ms": 455, "error_rate_pct": 8.0, "saturation_pct": 70, "availability_pct": 99.3}, "finops": ok_fin, "sec": ok_sec}, [R, M])
# ---------------- STANDARD: INCOMPLETE (marked missing) ----------------
case("NV-I01", "INCOMPLETE", "standard", {"sre": {"p95_latency_ms": 900, "error_rate_pct": None, "saturation_pct": 90, "availability_pct": 98.6, "_evidence": {"error_rate_pct": {"status": "missing"}}}, "deploy": {"pipeline_failed": False}},
     [M], ["get_error_rate_history", "get_latency_history"], "get_error_rate_history", {"sre": {"error_rate_pct": 13}})
case("NV-I02", "INCOMPLETE", "standard", {"deploy": {"restart_burst_count": 5, "restart_window_seconds": None, "_evidence": {"restart_window_seconds": {"status": "missing"}}}, "sre": {"p95_latency_ms": 850, "error_rate_pct": 11, "availability_pct": 98.5}},
     [R], ["get_restart_history", "get_deployment_history"], "get_restart_history", {"deploy": {"restart_window_seconds": 45}})
case("NV-I03", "INCOMPLETE", "standard", {"sec": {"critical_cves": None, "policy_violation": True, "_evidence": {"critical_cves": {"status": "missing"}}}, "deploy": {"pipeline_failed": True}},
     [P], ["get_vulnerability_details", "get_policy_gate_status"], "get_vulnerability_details", {"sec": {"critical_cves": 2}})
case("NV-I04", "INCOMPLETE", "standard", {"finops": {"cost_spike_pct": None, "hpa_scale_to": 15, "cpu_request_increase_pct": 58, "_evidence": {"cost_spike_pct": {"status": "missing"}}}, "sre": {"p95_latency_ms": 240, "error_rate_pct": 1.2}},
     [S, V], ["get_cost_proxy_history", "get_replica_history"], "get_cost_proxy_history", {"finops": {"cost_spike_pct": 40}})
case("NV-I05", "INCOMPLETE", "standard", {"sre": {"p95_latency_ms": 780, "error_rate_pct": 9.5, "saturation_pct": None, "_evidence": {"saturation_pct": {"status": "missing"}}}, "finops": {"cost_spike_pct": 26, "hpa_scale_to": 12}},
     [S], ["get_saturation_history", "get_cost_proxy_history"], "get_saturation_history", {"sre": {"saturation_pct": 93}})
case("NV-I06", "INCOMPLETE", "standard", {"deploy": {"restart_burst_count": 4, "restart_window_seconds": 50, "config_drift": None, "_evidence": {"config_drift": {"status": "missing"}}}, "sre": {"p95_latency_ms": 860, "error_rate_pct": 12, "availability_pct": 98.3}},
     [R], ["get_deployment_history", "get_latency_history"], "get_deployment_history", {"deploy": {"config_drift": True, "deployment_preceded_restarts": True}})
# ---------------- STANDARD: MISLEADING ----------------
case("NV-M01", "MISLEADING", "standard", {"finops": {"cost_spike_pct": 40, "hpa_scale_to": 15}, "sre": {"p95_latency_ms": 950, "error_rate_pct": 13, "saturation_pct": 94, "availability_pct": 97.9}, "deploy": {"pipeline_failed": False}, "sec": ok_sec}, [S])
case("NV-M02", "MISLEADING", "standard", {"sec": {"critical_cves": 1, "policy_violation": False, "iam_drift": False}, "sre": {"p95_latency_ms": 990, "error_rate_pct": 15, "saturation_pct": 93, "availability_pct": 96.8}, "deploy": {"pipeline_failed": False}, "finops": ok_fin}, [M])
case("NV-M03", "MISLEADING", "standard", {"finops": {"cost_spike_pct": 3, "hpa_scale_to": 13, "cpu_request_increase_pct": 4}, "sre": ok_sre, "deploy": {"pipeline_failed": False}, "sec": ok_sec}, [N, V])
case("NV-M04", "MISLEADING", "standard", {"deploy": {"restart_burst_count": 2, "restart_window_seconds": 100, "deployment_preceded_restarts": False}, "sre": {"p95_latency_ms": 870, "error_rate_pct": 11, "saturation_pct": 88, "availability_pct": 98.4}, "finops": ok_fin, "sec": ok_sec}, [M])
case("NV-M05", "MISLEADING", "standard", {"deploy": {"pipeline_failed": True, "config_drift": False}, "sec": {"critical_cves": 1, "policy_violation": True}, "sre": {"p95_latency_ms": 940, "error_rate_pct": 14, "saturation_pct": 92}, "finops": {"cost_spike_pct": 5}}, [P])
case("NV-M06", "MISLEADING", "standard", {"deploy": {"config_drift": True, "pipeline_failed": False, "artifact_mismatch": False}, "sre": ok_sre, "finops": ok_fin, "sec": ok_sec}, [B, R])

# ---------------- UNMARKED: decisive information absent, stale, or in another domain (no 'missing' flag) ----------------
U3 = ["get_restart_history", "get_error_rate_history", "get_cost_proxy_history"]
case("NV-U01", "UNMARKED", "absent", {"sre": {"p95_latency_ms": 880, "error_rate_pct": 11, "saturation_pct": 75, "availability_pct": 98.6}, "deploy": {"pipeline_failed": False}},
     [R], U3, "get_restart_history", {"deploy": {"restart_burst_count": 6, "restart_window_seconds": 35, "deployment_preceded_restarts": True}})
case("NV-U02", "UNMARKED", "absent", {"finops": {"cost_spike_pct": 20, "hpa_scale_to": 10}, "deploy": {"pipeline_failed": False}, "sec": ok_sec},
     [S], ["get_latency_history", "get_saturation_history", "get_replica_history"], "get_latency_history", {"sre": {"p95_latency_ms": 880, "error_rate_pct": 9, "saturation_pct": 93}})
case("NV-U03", "UNMARKED", "absent", {"deploy": {"pipeline_failed": True, "artifact_mismatch": False}, "sre": ok_sre, "finops": ok_fin},
     [P], ["get_policy_gate_status", "get_vulnerability_details", "get_pipeline_history"], "get_vulnerability_details", {"sec": {"critical_cves": 2, "policy_violation": True}})
case("NV-U04", "UNMARKED", "absent", {"sre": {"p95_latency_ms": 840, "error_rate_pct": 10.5, "saturation_pct": 90, "availability_pct": 98.8}, "deploy": {"pipeline_failed": False}, "sec": ok_sec},
     [S], ["get_replica_history", "get_cost_proxy_history", "get_error_rate_history"], "get_cost_proxy_history", {"finops": {"cost_spike_pct": 33, "hpa_scale_to": 13}})
case("NV-U05", "UNMARKED", "absent", {"deploy": {"config_drift": True, "artifact_mismatch": False, "pipeline_failed": False}, "finops": ok_fin, "sec": ok_sec},
     [R], ["get_latency_history", "get_error_rate_history", "get_replica_history"], "get_error_rate_history", {"sre": {"error_rate_pct": 12.5, "p95_latency_ms": 860}})
case("NV-U06", "UNMARKED", "absent", {"deploy": ok_dep, "sre": ok_sre, "finops": ok_fin},
     [P], ["get_vulnerability_details", "get_policy_gate_status", "get_cost_proxy_history"], "get_policy_gate_status", {"sec": {"policy_violation": True, "critical_cves": 1}})
case("NV-U07", "UNMARKED", "stale", {"deploy": {"config_drift": True, "pipeline_failed": False}, "sre": {"p95_latency_ms": 210, "error_rate_pct": 0.6, "availability_pct": 99.9, "_evidence": {"p95_latency_ms": st(180), "error_rate_pct": st(180), "availability_pct": st(180)}}},
     [R], ["get_latency_history", "get_error_rate_history", "get_cost_proxy_history"], "get_error_rate_history", {"sre": {"error_rate_pct": 13, "p95_latency_ms": 900, "availability_pct": 98.1}})
case("NV-U08", "UNMARKED", "stale", {"deploy": {"pipeline_failed": True, "artifact_mismatch": False}, "sec": {"critical_cves": 0, "policy_violation": False, "_evidence": {"critical_cves": st(10080), "policy_violation": st(10080)}}, "sre": ok_sre},
     [P], ["get_vulnerability_details", "get_policy_gate_status", "get_pipeline_history"], "get_vulnerability_details", {"sec": {"critical_cves": 2, "policy_violation": True}})
case("NV-U09", "UNMARKED", "stale", {"finops": {"cost_spike_pct": 5, "hpa_scale_to": 9, "_evidence": {"cost_spike_pct": st(2880)}}, "sre": {"p95_latency_ms": 860, "error_rate_pct": 11, "saturation_pct": 91}},
     [S], ["get_cost_proxy_history", "get_replica_history", "get_saturation_history"], "get_cost_proxy_history", {"finops": {"cost_spike_pct": 34, "hpa_scale_to": 13}})
case("NV-U10", "UNMARKED", "stale", {"sre": {"p95_latency_ms": 950, "error_rate_pct": 14, "saturation_pct": 95, "availability_pct": 97.0, "_evidence": {"p95_latency_ms": st(360), "error_rate_pct": st(360), "saturation_pct": st(360), "availability_pct": st(360)}}, "deploy": ok_dep, "finops": ok_fin},
     [N], ["get_latency_history", "get_error_rate_history", "get_restart_history"], "get_latency_history", {"sre": {"p95_latency_ms": 230, "error_rate_pct": 0.8, "saturation_pct": 45, "availability_pct": 99.95}})
case("NV-U11", "UNMARKED", "stale", {"sec": {"critical_cves": 0, "policy_violation": True, "_evidence": {"policy_violation": st(14400)}}, "deploy": ok_dep, "sre": ok_sre},
     [N], ["get_policy_gate_status", "get_vulnerability_details", "get_deployment_history"], "get_policy_gate_status", {"sec": {"policy_violation": False}})
case("NV-U12", "UNMARKED", "stale", {"finops": {"cost_spike_pct": 38, "hpa_scale_to": 15, "_evidence": {"cost_spike_pct": st(4320), "hpa_scale_to": st(4320)}}, "sre": ok_sre, "deploy": {"pipeline_failed": False}},
     [N], ["get_cost_proxy_history", "get_replica_history", "get_latency_history"], "get_cost_proxy_history", {"finops": {"cost_spike_pct": 3, "hpa_scale_to": 4}})
case("NV-U13", "UNMARKED", "cross_domain", {"sre": {"p95_latency_ms": 900, "error_rate_pct": 12, "saturation_pct": 70, "availability_pct": 98.4}, "deploy": {"pipeline_failed": False}},
     [R], ["get_deployment_history", "get_latency_history", "get_saturation_history"], "get_deployment_history", {"deploy": {"config_drift": True, "deployment_preceded_restarts": True, "restart_burst_count": 4, "restart_window_seconds": 50}})
case("NV-U14", "UNMARKED", "cross_domain", {"finops": {"cost_spike_pct": 36, "hpa_scale_to": 14}, "sre": ok_sre, "deploy": {"pipeline_failed": False}},
     [R, S], ["get_deployment_history", "get_cost_proxy_history", "get_replica_history"], "get_deployment_history", {"deploy": {"config_drift": True}})
case("NV-U15", "UNMARKED", "cross_domain", {"sre": {"p95_latency_ms": 880, "error_rate_pct": 13, "availability_pct": 98.2}, "sec": ok_sec},
     [R], ["get_vulnerability_details", "get_policy_gate_status", "get_error_rate_history", "get_deployment_history"], "get_deployment_history", {"deploy": {"artifact_mismatch": True, "deployment_preceded_restarts": True}})
case("NV-U16", "UNMARKED", "cross_domain", {"deploy": {"restart_burst_count": 3, "restart_window_seconds": 70}, "sre": {"p95_latency_ms": 260, "error_rate_pct": 2}},
     [R], ["get_saturation_history", "get_replica_history", "get_restart_history"], "get_saturation_history", {"sre": {"saturation_pct": 96, "availability_pct": 98.0}})
case("NV-U17", "UNMARKED", "cross_domain", {"finops": {"cost_spike_pct": 28, "hpa_scale_to": 12}, "deploy": {"pipeline_failed": False}, "sec": ok_sec},
     [S], ["get_latency_history", "get_saturation_history", "get_cost_proxy_history"], "get_saturation_history", {"sre": {"saturation_pct": 94, "p95_latency_ms": 820}})
case("NV-U18", "UNMARKED", "cross_domain", {"sre": {"availability_pct": 97.5}, "deploy": {"pipeline_failed": False}},
     [R], ["get_restart_history", "get_error_rate_history", "get_cost_proxy_history"], "get_restart_history", {"deploy": {"restart_burst_count": 6, "restart_window_seconds": 30, "deployment_preceded_restarts": True}})
# NV-U19, U20, U21, U24 are controls (fetching the relevant tool should not change the action); U22, U23 are decisive
case("NV-U19", "UNMARKED", "absent_control", {"deploy": {"pipeline_failed": True, "artifact_mismatch": False}, "sre": ok_sre, "finops": ok_fin},
     [B], ["get_policy_gate_status", "get_vulnerability_details", "get_pipeline_history"], "get_policy_gate_status", {"sec": {"policy_violation": False, "critical_cves": 0}})
case("NV-U20", "UNMARKED", "stale_control", {"sre": {"p95_latency_ms": 230, "error_rate_pct": 0.7, "availability_pct": 99.9, "_evidence": {"p95_latency_ms": st(30), "error_rate_pct": st(30)}}, "deploy": ok_dep, "finops": ok_fin},
     [N], ["get_latency_history", "get_error_rate_history", "get_cost_proxy_history"], "get_latency_history", {"sre": {"p95_latency_ms": 240, "error_rate_pct": 0.9}})
case("NV-U21", "UNMARKED", "cross_domain_control", {"sre": {"p95_latency_ms": 920, "error_rate_pct": 13, "saturation_pct": 93, "availability_pct": 97.8}, "finops": ok_fin},
     [M], ["get_deployment_history", "get_restart_history", "get_cost_proxy_history"], "get_deployment_history", {"deploy": {"config_drift": False, "deployment_preceded_restarts": False, "pipeline_failed": False}})
case("NV-U22", "UNMARKED", "absent", {"finops": {"cost_spike_pct": 31, "hpa_scale_to": 13, "cpu_request_increase_pct": 52}, "deploy": {"pipeline_failed": True}, "sre": ok_sre},
     [P], ["get_vulnerability_details", "get_policy_gate_status", "get_replica_history"], "get_vulnerability_details", {"sec": {"critical_cves": 2, "policy_violation": True}})
case("NV-U23", "UNMARKED", "stale", {"deploy": {"config_drift": False, "_evidence": {"config_drift": st(2880)}}, "sre": {"p95_latency_ms": 870, "error_rate_pct": 11.5, "saturation_pct": 80}},
     [R], ["get_deployment_history", "get_latency_history", "get_saturation_history"], "get_deployment_history", {"deploy": {"config_drift": True, "deployment_preceded_restarts": True}})
case("NV-U24", "UNMARKED", "cross_domain_control", {"sec": {"critical_cves": 2, "policy_violation": True}, "sre": ok_sre, "finops": ok_fin},
     [P], ["get_pipeline_history", "get_deployment_history", "get_policy_gate_status"], "get_pipeline_history", {"deploy": {"pipeline_failed": True}})

out = Path(__file__).parent
cases = [c for c, _ in C]
ids = [c["id"] for c in cases]
assert len(ids) == len(set(ids)) == 48
(out / "cases_new_v1.yaml").write_text("# FROZEN BEFORE EXPERT LABELLING AND BEFORE ANY RUN. See PROTOCOL.md.\n" + yaml.safe_dump({"version": 1, "n_cases": 48, "cases": cases}, sort_keys=False, width=200))
(out / "author_labels.yaml").write_text("# Author-proposed admissible actions, sealed with the cases; NOT the primary key for new cases.\n" + yaml.safe_dump({c["id"]: a for c, a in C}, sort_keys=True, width=200))
print("wrote", len(cases))
