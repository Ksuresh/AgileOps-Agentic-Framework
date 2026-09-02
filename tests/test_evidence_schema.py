from agents.sre import SREAgent
from agents.finops import FinOpsAgent
from agents.devops import DevOpsAgent
from evidence.schema import get_evidence


def test_missing_numeric_zero_is_not_usable():
    block={"p95_latency_ms":0.0,"_evidence":{"p95_latency_ms":{"status":"missing","source":"not instrumented"}}}
    ev=get_evidence(block,"p95_latency_ms")
    assert ev.status=="missing"
    assert not ev.usable


def test_sre_missing_fields_do_not_become_healthy_measurements():
    telemetry={"sre":{"availability_pct":99.9,"_evidence":{"p95_latency_ms":{"status":"missing"},"error_rate_pct":{"status":"missing"},"saturation_pct":{"status":"missing"},"availability_pct":{"status":"proxy","source":"process state"}}}}
    out=SREAgent().infer(telemetry)
    assert "No material reliability anomaly" in out.claim
    assert any("available SRE evidence" in x for x in out.evidence)


def test_sre_proxy_failure_is_labelled_proxy():
    telemetry={"sre":{"availability_pct":0.0,"_evidence":{"availability_pct":{"status":"proxy","source":"process state"}}}}
    out=SREAgent().infer(telemetry)
    assert any("[proxy]" in x for x in out.evidence)


def test_finops_footprint_proxy_is_not_called_measured_cost():
    telemetry={"finops":{"cost_spike_pct":40.0,"_evidence":{"cost_spike_pct":{"status":"proxy","source":"container footprint"}}}}
    out=FinOpsAgent().infer(telemetry)
    assert any("[proxy]" in x for x in out.evidence)


def test_legacy_controlled_devops_values_remain_supported():
    telemetry={"deploy":{"pipeline_failed":True,"config_drift":True,"rollback_marker":False,"artifact_mismatch":False,"restart_loops":0}}
    out=DevOpsAgent().infer(telemetry)
    assert "release or configuration issue" in out.claim or "primary operational cause" in out.claim
