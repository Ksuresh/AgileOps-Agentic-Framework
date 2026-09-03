from __future__ import annotations

"""Run second-batch held-out Sock Shop compound cases with direct runtime evidence.

The runner never reads oracle actions to choose behavior. It supports two frozen
intervention patterns:
- HRT-06: live HTTP load while the front-end undergoes a restart burst.
- HRT-07: scale-out plus strict CPU quota under live HTTP load.

Both cases preserve direct HTTP observations. HRT-06 additionally preserves
Docker State.StartedAt transitions; HRT-07 preserves the applied CPU quota and
scale parameters in intervention metadata. The AAF policy is not modified by
this runner.
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

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = Path(__file__).with_name("interventions.yaml")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run(cmd: list[str], *, check: bool = True, capture: bool = False) -> subprocess.CompletedProcess:
    print("+", " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=ROOT, text=True, check=check, capture_output=capture)


def load_case(case_id: str) -> dict:
    data = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    for case in data["cases"]:
        if case["id"] == case_id:
            if case_id not in {"HRT-06", "HRT-07"}:
                raise SystemExit("This runner is restricted to HRT-06 and HRT-07")
            return case
    raise SystemExit(f"Unknown case: {case_id}")


def resolve_load_url(compose_file: str) -> str:
    cp = run(["docker", "compose", "-f", compose_file, "port", "edge-router", "80"], capture=True)
    lines = [line.strip() for line in cp.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("edge-router port 80 is not published")
    port = lines[0].rsplit(":", 1)[-1]
    return f"http://127.0.0.1:{port}/catalogue"


def preflight(url: str) -> int:
    last = "unknown"
    for _ in range(15):
        try:
            with urllib.request.urlopen(url, timeout=5.0) as response:
                status = int(response.status)
                response.read(512)
                if 200 <= status < 400:
                    return status
                last = f"HTTP {status}"
        except Exception as exc:
            last = type(exc).__name__
        time.sleep(2)
    raise RuntimeError(f"load endpoint failed preflight: {url} ({last})")


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    xs = sorted(values)
    idx = max(0, min(len(xs) - 1, math.ceil(q * len(xs)) - 1))
    return xs[idx]


def worker(url: str, stop: threading.Event, state: dict, lock: threading.Lock) -> None:
    while not stop.is_set():
        started = time.perf_counter()
        status = None
        try:
            with urllib.request.urlopen(url, timeout=3.0) as response:
                response.read(1024)
                status = int(response.status)
        except urllib.error.HTTPError as exc:
            status = int(exc.code)
        except Exception:
            status = None
        latency_ms = (time.perf_counter() - started) * 1000.0
        ok = status is not None and 200 <= status < 400
        with lock:
            state["requests"] += 1
            state["successes" if ok else "failures"] += 1
            state["latencies_ms"].append(latency_ms)
            if status is not None:
                key = str(status)
                state["status_counts"][key] = state["status_counts"].get(key, 0) + 1


def service_started_at(compose_file: str, service: str) -> str | None:
    cp = run(["docker", "compose", "-f", compose_file, "ps", "-q", service], capture=True)
    ids = [x.strip() for x in cp.stdout.splitlines() if x.strip()]
    if not ids:
        return None
    insp = run(["docker", "inspect", "-f", "{{.State.StartedAt}}", ids[0]], capture=True)
    return insp.stdout.strip() or None


def latest_artifact(case_id: str, repetition: int) -> Path | None:
    base = ROOT / "runtime_validation" / "artifacts" / case_id / f"rep-{repetition}"
    dirs = sorted([p for p in base.glob("*") if p.is_dir()]) if base.exists() else []
    return dirs[-1] if dirs else None


def write_metadata(case: dict, repetition: int, payload: dict) -> None:
    out = ROOT / "runtime_validation" / "run_metadata" / case["id"] / f"rep-{repetition}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "compound_intervention.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def apply_cpu_quota(compose_file: str, service: str, cpus: float) -> list[str]:
    cp = run(["docker", "compose", "-f", compose_file, "ps", "-q", service], capture=True)
    ids = [x.strip() for x in cp.stdout.splitlines() if x.strip()]
    if not ids:
        raise RuntimeError(f"No active containers for {service}")
    for cid in ids:
        run(["docker", "update", "--cpus", str(cpus), cid])
    return ids


def reset_cpu_quota(container_ids: list[str]) -> None:
    for cid in container_ids:
        run(["docker", "update", "--cpus", "0", cid], check=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("case_id", choices=["HRT-06", "HRT-07"])
    ap.add_argument("--repetition", type=int, required=True)
    ap.add_argument("--compose-file", required=True)
    args = ap.parse_args()

    case = load_case(args.case_id)
    if not 1 <= args.repetition <= int(case.get("repetitions", 1)):
        raise SystemExit("Repetition outside frozen manifest range")
    p = case.get("parameters", {}) or {}
    compose = args.compose_file

    run(["docker", "compose", "-f", compose, "up", "-d", "--remove-orphans", "--scale", "front-end=1"])
    time.sleep(8)

    cpu_limited_ids: list[str] = []
    temporal = None
    if args.case_id == "HRT-07":
        replicas = int(p.get("replicas", 6))
        run(["docker", "compose", "-f", compose, "up", "-d", "--scale", f"front-end={replicas}"])
        time.sleep(8)
        cpu_limited_ids = apply_cpu_quota(compose, "front-end", float(p.get("cpus", 0.10)))

    url = resolve_load_url(compose)
    preflight_status = preflight(url)
    concurrency = int(p.get("concurrency", 100))
    duration = int(p.get("duration_seconds", 90))
    stop = threading.Event()
    state = {"requests": 0, "successes": 0, "failures": 0, "latencies_ms": [], "status_counts": {}}
    lock = threading.Lock()
    threads = [threading.Thread(target=worker, args=(url, stop, state, lock), daemon=True) for _ in range(concurrency)]
    started = utc_now()
    for thread in threads:
        thread.start()

    if args.case_id == "HRT-06":
        time.sleep(float(p.get("load_warmup_seconds", 8)))
        service = case.get("target_service", "front-end")
        previous = service_started_at(compose, service)
        samples = [previous] if previous else []
        observed = 0
        for _ in range(int(p.get("restart_count", 4))):
            run(["docker", "compose", "-f", compose, "restart", service])
            time.sleep(float(p.get("restart_interval_seconds", 15)))
            current = service_started_at(compose, service)
            if current:
                samples.append(current)
                if previous is not None and current != previous:
                    observed += 1
                previous = current
        temporal = {
            "case_id": args.case_id,
            "repetition": args.repetition,
            "observed_restart_events": observed,
            "started_at_samples": samples,
            "source": "restart transitions observed during concurrent HTTP load via Docker State.StartedAt",
            "captured_utc": utc_now(),
        }

    elapsed = 0.0
    # Keep load active long enough to capture the resource/runtime snapshot.
    if args.case_id == "HRT-06":
        elapsed = float(p.get("load_warmup_seconds", 8)) + int(p.get("restart_count", 4)) * float(p.get("restart_interval_seconds", 15))
    else:
        time.sleep(min(20.0, duration / 3.0))
        elapsed = min(20.0, duration / 3.0)

    collector = ROOT / "runtime_validation" / "collect_runtime_artifacts.sh"
    run(["bash", str(collector), args.case_id, str(args.repetition), compose])
    time.sleep(max(0.0, duration - elapsed))
    stop.set()
    for thread in threads:
        thread.join(timeout=4)
    finished = utc_now()

    artifact = latest_artifact(args.case_id, args.repetition)
    if artifact is None:
        raise RuntimeError("Runtime artifact was not created")
    with lock:
        latencies = list(state["latencies_ms"])
        requests = int(state["requests"])
        successes = int(state["successes"])
        failures = int(state["failures"])
        status_counts = dict(state["status_counts"])
    if requests == 0 or successes == 0:
        raise RuntimeError("No successful HTTP traffic observed; refusing invalid transport evidence")

    load_obs = {
        "case_id": args.case_id,
        "repetition": args.repetition,
        "started_utc": started,
        "finished_utc": finished,
        "url": url,
        "preflight_status": preflight_status,
        "concurrency": concurrency,
        "duration_seconds": duration,
        "requests": requests,
        "successes": successes,
        "failures": failures,
        "error_rate_pct": 100.0 * failures / requests,
        "latency_p50_ms": percentile(latencies, 0.50),
        "latency_p95_ms": percentile(latencies, 0.95),
        "latency_p99_ms": percentile(latencies, 0.99),
        "status_counts": status_counts,
        "source": "direct concurrent HTTP measurements through Sock Shop edge-router /catalogue",
    }
    (artifact / "load_observation.json").write_text(json.dumps(load_obs, indent=2), encoding="utf-8")
    if temporal is not None:
        (artifact / "temporal_process_observation.json").write_text(json.dumps(temporal, indent=2), encoding="utf-8")

    write_metadata(case, args.repetition, {
        "case_id": args.case_id,
        "repetition": args.repetition,
        "parameters": p,
        "load_observation": load_obs,
        "temporal_process_observation": temporal,
        "cpu_limited_container_ids": cpu_limited_ids,
        "captured_utc": utc_now(),
    })

    if cpu_limited_ids:
        reset_cpu_quota(cpu_limited_ids)
    run(["docker", "compose", "-f", compose, "up", "-d", "--scale", "front-end=1"])
    print(json.dumps({"load": load_obs, "temporal": temporal}, indent=2))


if __name__ == "__main__":
    main()
