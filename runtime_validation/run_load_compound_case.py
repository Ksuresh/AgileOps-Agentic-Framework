from __future__ import annotations

"""Execute Sock Shop load cases and preserve request-level runtime evidence.

Supports frozen RT-11 and RT-12 only. The runner does not read oracle labels to
drive behavior. It records measured HTTP latency/error observations while load
is active and stores only derived aggregates plus request counts in the artifact.
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
    return subprocess.run(cmd, cwd=ROOT, text=True, check=check, capture_output=capture)


def load_case(case_id: str) -> dict:
    data = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    for case in data["cases"]:
        if case["id"] == case_id:
            if case_id not in {"RT-11", "RT-12"}:
                raise SystemExit("This executor is restricted to RT-11 and RT-12")
            return case
    raise SystemExit(f"Unknown case: {case_id}")


def resolve_frontend_url(compose_file: str) -> str:
    cp = run(["docker", "compose", "-f", compose_file, "port", "front-end", "8079"], capture=True)
    value = cp.stdout.strip().splitlines()[0]
    port = value.rsplit(":", 1)[-1]
    # /catalogue exercises a backend dependency through the front-end rather than
    # measuring only static home-page delivery.
    return f"http://127.0.0.1:{port}/catalogue"


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
                state["status_counts"][str(status)] = state["status_counts"].get(str(status), 0) + 1


def latest_artifact(case_id: str, repetition: int) -> Path | None:
    base = ROOT / "runtime_validation" / "artifacts" / case_id / f"rep-{repetition}"
    dirs = sorted([p for p in base.glob("*") if p.is_dir()]) if base.exists() else []
    return dirs[-1] if dirs else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("case_id", choices=["RT-11", "RT-12"])
    ap.add_argument("--repetition", type=int, required=True)
    ap.add_argument("--compose-file", required=True)
    ap.add_argument("--duration-seconds", type=int)
    ap.add_argument("--concurrency", type=int)
    args = ap.parse_args()

    case = load_case(args.case_id)
    if not 1 <= args.repetition <= int(case.get("repetitions", 1)):
        raise SystemExit("Repetition outside frozen manifest range")

    compose = args.compose_file
    params = case.get("parameters", {}) or {}
    # Use frozen manifest values unless an explicit override is provided.
    concurrency = int(args.concurrency if args.concurrency is not None else params.get("concurrency", 50))
    duration = int(args.duration_seconds if args.duration_seconds is not None else params.get("duration_seconds", 120))

    run(["docker", "compose", "-f", compose, "up", "-d", "--scale", "front-end=1"])
    time.sleep(5)
    if args.case_id == "RT-12":
        replicas = int(params.get("replicas", 8))
        run(["docker", "compose", "-f", compose, "up", "-d", "--scale", f"front-end={replicas}"])
        time.sleep(5)

    url = resolve_frontend_url(compose)
    stop = threading.Event()
    state = {"requests": 0, "successes": 0, "failures": 0, "latencies_ms": [], "status_counts": {}}
    lock = threading.Lock()
    threads = [threading.Thread(target=worker, args=(url, stop, state, lock), daemon=True) for _ in range(concurrency)]

    started = utc_now()
    for thread in threads:
        thread.start()
    warmup = max(5, min(20, duration // 4))
    time.sleep(warmup)

    collector = ROOT / "runtime_validation" / "collect_runtime_artifacts.sh"
    run(["bash", str(collector), args.case_id, str(args.repetition), compose])

    time.sleep(max(0, duration - warmup))
    stop.set()
    for thread in threads:
        thread.join(timeout=4)
    finished = utc_now()

    artifact = latest_artifact(args.case_id, args.repetition)
    if artifact is None:
        raise SystemExit("Runtime artifact was not created")

    with lock:
        latencies = list(state["latencies_ms"])
        requests = int(state["requests"])
        failures = int(state["failures"])
        successes = int(state["successes"])
        status_counts = dict(state["status_counts"])

    observation = {
        "case_id": args.case_id,
        "repetition": args.repetition,
        "started_utc": started,
        "finished_utc": finished,
        "url": url,
        "concurrency": concurrency,
        "duration_seconds": duration,
        "requests": requests,
        "successes": successes,
        "failures": failures,
        "error_rate_pct": (100.0 * failures / requests) if requests else None,
        "latency_p50_ms": percentile(latencies, 0.50),
        "latency_p95_ms": percentile(latencies, 0.95),
        "latency_p99_ms": percentile(latencies, 0.99),
        "status_counts": status_counts,
        "source": "direct concurrent HTTP measurements against Sock Shop front-end /catalogue",
    }
    (artifact / "load_observation.json").write_text(json.dumps(observation, indent=2), encoding="utf-8")

    run(["docker", "compose", "-f", compose, "up", "-d", "--scale", "front-end=1"])
    print(json.dumps(observation, indent=2))


if __name__ == "__main__":
    main()
