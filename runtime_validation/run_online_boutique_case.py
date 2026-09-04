from __future__ import annotations

"""Run one frozen Google Online Boutique scenario and preserve evidence.

The runner does not read oracle labels when applying an intervention. It records
benchmark-specific Kubernetes/HTTP evidence and then passes only that evidence
to the unchanged AAF decision path.
"""

import argparse
import json
import math
import subprocess
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import yaml

from orchestrator.cross_domain import apply_interaction_policy
from orchestrator.decision_baselines import choose_dominant_domain_action
from orchestrator.utility import choose_action_details

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = Path(__file__).with_name("interventions_online_boutique.yaml")
OUT = ROOT / "results_online_boutique_runtime"
HTTP_TIMEOUT = 2.0


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run(cmd: list[str], *, capture: bool = False, check: bool = True, timeout: int = 180):
    print("+", " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=ROOT, text=True, capture_output=capture, check=check, timeout=timeout)


def load_case(case_id: str) -> dict:
    data = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    for case in data["cases"]:
        if case["id"] == case_id:
            return case
    raise SystemExit(f"unknown Online Boutique case: {case_id}")


def wait_frontend(replicas: int = 1, timeout_s: int = 180) -> None:
    if replicas <= 0:
        return
    run(["kubectl", "rollout", "status", "deployment/frontend", f"--timeout={timeout_s}s"], timeout=timeout_s + 30)


def reset_frontend() -> None:
    # Restore the canonical deployment shape before each independent scenario.
    run(["kubectl", "scale", "deployment/frontend", "--replicas=1"])
    run(["kubectl", "set", "resources", "deployment/frontend", "--requests=cpu=100m,memory=64Mi", "--limits=cpu=200m,memory=128Mi"])
    run(["kubectl", "set", "env", "deployment/frontend", "AAF_OB_TEST_POLICY_MARKER-"] , check=False)
    wait_frontend(1)


def get_replicas() -> int:
    cp = run(["kubectl", "get", "deployment/frontend", "-o", "jsonpath={.spec.replicas}"], capture=True)
    return int(cp.stdout.strip() or "0")


def get_pod_uids() -> list[str]:
    cp = run(["kubectl", "get", "pods", "-l", "app=frontend", "-o", "jsonpath={range .items[*]}{.metadata.uid}{'\\n'}{end}"], capture=True)
    return sorted(x.strip() for x in cp.stdout.splitlines() if x.strip())


def marker_present() -> bool:
    cp = run(["kubectl", "get", "deployment/frontend", "-o", "json"], capture=True)
    obj = json.loads(cp.stdout)
    for c in obj.get("spec", {}).get("template", {}).get("spec", {}).get("containers", []):
        for item in c.get("env", []) or []:
            if item.get("name") == "AAF_OB_TEST_POLICY_MARKER" and item.get("value") == "exposed_test_secret":
                return True
    return False


def worker(url: str, stop: threading.Event, deadline: float, state: dict, lock: threading.Lock) -> None:
    while not stop.is_set() and time.monotonic() < deadline:
        t0 = time.perf_counter()
        status = None
        try:
            with urllib.request.urlopen(url, timeout=HTTP_TIMEOUT) as r:
                r.read(512)
                status = int(r.status)
        except urllib.error.HTTPError as e:
            status = int(e.code)
        except Exception:
            status = None
        ms = (time.perf_counter() - t0) * 1000.0
        ok = status is not None and 200 <= status < 400
        with lock:
            state["requests"] += 1
            state["successes" if ok else "failures"] += 1
            state["latencies"].append(ms)


def percentile(xs: list[float], q: float) -> float | None:
    if not xs:
        return None
    ys = sorted(xs)
    return ys[max(0, min(len(ys) - 1, math.ceil(q * len(ys)) - 1))]


def measure_http(url: str, duration_s: int, concurrency: int) -> dict:
    stop = threading.Event()
    lock = threading.Lock()
    state = {"requests": 0, "successes": 0, "failures": 0, "latencies": []}
    deadline = time.monotonic() + duration_s
    threads = [threading.Thread(target=worker, args=(url, stop, deadline, state, lock), daemon=True) for _ in range(concurrency)]
    start = utc_now()
    for t in threads:
        t.start()
    time.sleep(duration_s)
    stop.set()
    for t in threads:
        t.join(timeout=3)
    with lock:
        req = state["requests"]
        suc = state["successes"]
        fail = state["failures"]
        lat = list(state["latencies"])
    if req == 0:
        raise RuntimeError("no HTTP request attempts recorded")
    return {
        "started_utc": start,
        "finished_utc": utc_now(),
        "requests": req,
        "successes": suc,
        "failures": fail,
        "error_rate_pct": 100.0 * fail / req,
        "availability_pct": 100.0 * suc / req,
        "latency_p50_ms": percentile(lat, 0.50),
        "latency_p95_ms": percentile(lat, 0.95),
        "latency_p99_ms": percentile(lat, 0.99),
        "source": "direct HTTP measurements through kubectl port-forward to Online Boutique frontend",
    }


def evidence_meta(status: str, source: str, note: str = "") -> dict:
    return {"status": status, "source": source, "note": note}


def build_telemetry(case: dict, http: dict | None, restart_count: int | None, restart_window_s: float | None, replicas: int, marker: bool) -> dict:
    deploy = {"_evidence": {}}
    sre = {"_evidence": {}}
    finops = {"_evidence": {}}
    sec = {"_evidence": {}}

    if restart_count is not None:
        deploy["restart_burst_count"] = restart_count
        deploy["restart_window_seconds"] = restart_window_s
        deploy["restart_loops"] = restart_count
        deploy["_evidence"]["restart_burst_count"] = evidence_meta("measured", "observed Kubernetes frontend pod-UID transitions")
        deploy["_evidence"]["restart_window_seconds"] = evidence_meta("measured", "wall-clock interval spanning observed restart operations")
        deploy["_evidence"]["restart_loops"] = evidence_meta("measured", "observed restart operations")

    if marker:
        deploy["config_drift"] = True
        deploy["_evidence"]["config_drift"] = evidence_meta("measured", "observed Kubernetes Deployment environment", "test-only marker differs from frozen healthy deployment")
        sec["policy_violation"] = True
        sec["_evidence"]["policy_violation"] = evidence_meta("measured", "observed Kubernetes Deployment environment policy check", "test-only secret-like marker violates prespecified runtime-configuration policy")

    if http is not None:
        for field, src in (("p95_latency_ms", "latency_p95_ms"), ("error_rate_pct", "error_rate_pct"), ("availability_pct", "availability_pct")):
            sre[field] = http[src]
            sre["_evidence"][field] = evidence_meta("measured", http["source"])

    # Replica expansion is deliberately represented as a resource-footprint proxy,
    # not as monetary cloud cost.
    if replicas != 1:
        proxy = 100.0 * (replicas - 1) / 1.0
        finops["cost_spike_pct"] = proxy
        finops["_evidence"]["cost_spike_pct"] = evidence_meta("proxy", "observed Kubernetes frontend replica count relative to one-replica healthy baseline", "resource-footprint proxy; not monetary cost")

    return {"deploy": deploy, "sre": sre, "finops": finops, "sec": sec}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("case_id")
    ap.add_argument("--url", default="http://127.0.0.1:8080/")
    args = ap.parse_args()
    case = load_case(args.case_id)
    p = case.get("parameters", {}) or {}
    kind = case["intervention"]

    reset_frontend()
    restart_count = None
    restart_window_s = None
    http = None

    try:
        if kind == "stop_service":
            run(["kubectl", "scale", "deployment/frontend", "--replicas=0"])
            time.sleep(5)
            http = measure_http(args.url, 12, 8)

        elif kind == "scale_service":
            run(["kubectl", "scale", "deployment/frontend", f"--replicas={int(p['replicas'])}"])
            wait_frontend(int(p["replicas"]))
            http = measure_http(args.url, 12, 8)

        elif kind in {"restart_burst", "restart_burst_load"}:
            start = time.monotonic()
            before = get_pod_uids()
            observed = 0
            load_thread = None
            load_result: dict = {}
            if kind == "restart_burst_load":
                def do_load():
                    load_result["http"] = measure_http(args.url, int(p["duration_seconds"]), int(p["concurrency"]))
                load_thread = threading.Thread(target=do_load, daemon=True)
                load_thread.start()
                time.sleep(5)
            for _ in range(int(p["restart_count"])):
                run(["kubectl", "rollout", "restart", "deployment/frontend"])
                time.sleep(float(p["restart_interval_seconds"]))
                now = get_pod_uids()
                if now != before:
                    observed += 1
                before = now
            restart_count = observed
            restart_window_s = time.monotonic() - start
            if load_thread is not None:
                load_thread.join(timeout=int(p["duration_seconds"]) + 20)
                http = load_result.get("http")
            else:
                wait_frontend(1)
                http = measure_http(args.url, 12, 8)

        elif kind == "config_security":
            run(["kubectl", "set", "env", "deployment/frontend", "AAF_OB_TEST_POLICY_MARKER=exposed_test_secret"])
            wait_frontend(1)
            http = measure_http(args.url, 12, 8)

        elif kind in {"cpu_load", "scale_cpu_load"}:
            if kind == "scale_cpu_load":
                run(["kubectl", "scale", "deployment/frontend", f"--replicas={int(p['replicas'])}"])
            cpu = str(p["cpu_limit"])
            run(["kubectl", "set", "resources", "deployment/frontend", f"--requests=cpu={cpu},memory=64Mi", f"--limits=cpu={cpu},memory=128Mi"])
            wait_frontend(int(p.get("replicas", 1)))
            http = measure_http(args.url, int(p["duration_seconds"]), int(p["concurrency"]))

        elif kind == "healthy":
            http = measure_http(args.url, 12, 8)

        else:
            raise RuntimeError(f"unsupported intervention: {kind}")

        replicas = get_replicas()
        marker = marker_present()
        telemetry = build_telemetry(case, http, restart_count, restart_window_s, replicas, marker)
        oracle = set(case["admissible_actions"])

        baseline_action, baseline_severity, baseline_domain = choose_dominant_domain_action(telemetry)
        no_int = choose_action_details(telemetry, (0.4, 0.3, 0.3))["selected_action"]
        full = apply_interaction_policy(telemetry, no_int)
        full_action = full["selected_action"]

        row = {
            "case_id": case["id"],
            "title": case["title"],
            "oracle_actions": sorted(oracle),
            "telemetry": telemetry,
            "http": http,
            "observed_replicas": replicas,
            "observed_restart_burst_count": restart_count,
            "observed_restart_window_seconds": restart_window_s,
            "observed_security_marker": marker,
            "dominant_domain_baseline_action": baseline_action,
            "dominant_domain_baseline_match": baseline_action in oracle,
            "dominant_domain": baseline_domain,
            "dominant_domain_severity": baseline_severity,
            "aaf_no_interaction_action": no_int,
            "aaf_no_interaction_match": no_int in oracle,
            "aaf_full_action": full_action,
            "aaf_full_match": full_action in oracle,
            "interaction_applied": bool(full["interaction_applied"]),
            "dominant_interaction": None if full["interaction_state"].get("dominant_interaction") is None else full["interaction_state"]["dominant_interaction"]["name"],
            "captured_utc": utc_now(),
        }
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / f"{case['id']}.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
        print(json.dumps({k: row[k] for k in ["case_id", "dominant_domain_baseline_action", "aaf_no_interaction_action", "aaf_full_action", "aaf_full_match", "dominant_interaction"]}, indent=2))
    finally:
        reset_frontend()


if __name__ == "__main__":
    main()
