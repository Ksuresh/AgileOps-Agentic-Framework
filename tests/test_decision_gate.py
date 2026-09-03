from __future__ import annotations

import json

from orchestrator.decision_gate import materiality_gate
from orchestrator.utility import choose_action_details
from runtime_validation.evidence_adapter import derive_observables, to_aaf_telemetry


def _ev(status: str, source: str = "test"):
    return {"status": status, "source": source, "note": ""}


def healthy_runtime_telemetry():
    return {
        "deploy": {"restart_loops": 0, "_evidence": {"restart_loops": _ev("measured")}},
        "sre": {
            "saturation_pct": 15.0,
            "availability_pct": 99.9,
            "_evidence": {
                "saturation_pct": _ev("proxy"),
                "availability_pct": _ev("proxy"),
                "p95_latency_ms": _ev("missing"),
                "error_rate_pct": _ev("missing"),
            },
        },
        "finops": {"cost_spike_pct": 0.0, "_evidence": {"cost_spike_pct": _ev("proxy")}},
        "sec": {"_evidence": {"critical_cves": _ev("missing")}},
    }


def test_healthy_evidence_abstains_instead_of_forcing_action():
    telemetry = healthy_runtime_telemetry()
    gate = materiality_gate(telemetry)
    assert gate["decision"] == "observe"
    decision = choose_action_details(telemetry, (0.4, 0.3, 0.3))
    assert decision["selected_action"] == "No action (observe)"
    assert decision["eligible_actions"] == ["No action (observe)"]


def test_material_reliability_signal_allows_only_reliability_actions():
    telemetry = healthy_runtime_telemetry()
    telemetry["sre"]["availability_pct"] = 0.0
    gate = materiality_gate(telemetry)
    assert gate["decision"] == "act"
    assert gate["active_domains"] == ["reliability"]
    decision = choose_action_details(telemetry, (0.4, 0.3, 0.3))
    assert decision["selected_action"] == "Mitigate and monitor"
    assert decision["eligible_actions"] == ["Mitigate and monitor"]


def test_clean_one_shot_container_exit_is_not_availability_failure(tmp_path):
    inspect_dir = tmp_path / "inspect"
    inspect_dir.mkdir()
    payload = [{
        "State": {"Status": "exited", "ExitCode": 0, "OOMKilled": False},
        "HostConfig": {"RestartPolicy": {"Name": "no"}},
        "RestartCount": 0,
    }]
    (inspect_dir / "user-sim-1.json").write_text(json.dumps(payload), encoding="utf-8")
    (tmp_path / "compose_ps.txt").write_text("NAME  STATUS\nuser-sim-1  Exited (0)\n", encoding="utf-8")
    (tmp_path / "docker_stats.txt").write_text("", encoding="utf-8")
    obs = derive_observables(tmp_path)
    assert obs["availability_proxy"]["expected_completed_rows"] == 1
    assert obs["availability_proxy"]["exited_rows"] == 0
    telemetry = to_aaf_telemetry(obs)
    assert telemetry["sre"]["availability_pct"] == 99.9


def test_stopped_continuous_service_remains_availability_failure(tmp_path):
    inspect_dir = tmp_path / "inspect"
    inspect_dir.mkdir()
    payload = [{
        "State": {"Status": "exited", "ExitCode": 0, "OOMKilled": False},
        "HostConfig": {"RestartPolicy": {"Name": "always"}},
        "RestartCount": 0,
    }]
    (inspect_dir / "catalogue-1.json").write_text(json.dumps(payload), encoding="utf-8")
    (tmp_path / "compose_ps.txt").write_text("NAME  STATUS\ncatalogue-1  Exited (0)\n", encoding="utf-8")
    (tmp_path / "docker_stats.txt").write_text("", encoding="utf-8")
    obs = derive_observables(tmp_path)
    assert obs["availability_proxy"]["exited_rows"] == 1
    telemetry = to_aaf_telemetry(obs)
    assert telemetry["sre"]["availability_pct"] == 0.0


def test_stopped_continuous_service_detected_from_compose_when_stopped_container_is_not_inspected(tmp_path):
    """Mirrors RT-02: compose PS retains the stopped service even if ps -q does not."""
    (tmp_path / "inspect").mkdir()
    (tmp_path / "compose_ps.txt").write_text(
        "NAME                IMAGE                         COMMAND  SERVICE    CREATED  STATUS                     PORTS\n"
        "sock-catalogue-1    weaveworksdemos/catalogue    x        catalogue  1m ago   Exited (0) 10 seconds ago\n"
        "sock-user-sim-1     weaveworksdemos/load-test    x        user-sim   1m ago   Exited (0) 5 seconds ago\n",
        encoding="utf-8",
    )
    (tmp_path / "compose_resolved.yaml").write_text(
        "services:\n"
        "  catalogue:\n"
        "    image: weaveworksdemos/catalogue\n"
        "    restart: always\n"
        "  user-sim:\n"
        "    image: weaveworksdemos/load-test\n",
        encoding="utf-8",
    )
    (tmp_path / "docker_stats.txt").write_text("", encoding="utf-8")

    obs = derive_observables(tmp_path)
    assert obs["availability_proxy"]["exited_rows"] == 1
    assert obs["availability_proxy"]["expected_completed_rows"] == 1
    telemetry = to_aaf_telemetry(obs)
    assert telemetry["sre"]["availability_pct"] == 0.0
    gate = materiality_gate(telemetry)
    assert gate["decision"] == "act"
    assert gate["active_domains"] == ["reliability"]
    decision = choose_action_details(telemetry, (0.4, 0.3, 0.3))
    assert decision["selected_action"] == "Mitigate and monitor"
