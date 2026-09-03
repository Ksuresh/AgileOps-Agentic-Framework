from __future__ import annotations

"""Execute Sock Shop runtime load cases while collecting evidence during load.

This runner supports only the frozen RT-11 and RT-12 interventions. It uses the
live Sock Shop front-end, generates concurrent HTTP requests, and invokes the
existing raw-artifact collector while load is still active. No oracle value is
read to drive the intervention.
"""

import argparse
import json
import subprocess
import threading
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = Path(__file__).with_name("interventions.yaml")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run(cmd: list[str], *, check: bool = True, capture: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        cwd=ROOT,
        text=True,
        check=check,
        capture_output=capture,
    )


def load_case(case_id: str) -> dict:
    data = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    for case in data["cases"]:
        if case["id"] == case_id:
            if case_id not in {"RT-11", "RT-12"}:
                raise SystemExit("This executor is restricted to RT-11 and RT-12")
            return case
    raise SystemExit(f"Unknown case: {case_id}")


def resolve_frontend_url(compose_file: str) -> str:
    cp = run(
        ["docker", "compose", "-f", compose_file, "port", "front-end", "8079"],
        capture=True,
    )
    value = cp.stdout.strip().splitlines()[0]
    port = value.rsplit(":", 1)[-1]
    return f"http://127.0.0.1:{port}/"


def worker(url: str, stop: threading.Event, counters: dict[str, int], lock: threading.Lock) -> None:
    while not stop.is_set():
        ok = False
        try:
            with urllib.request.urlopen(url, timeout=2.0) as response:
                response.read(512)
                ok = 200 <= int(response.status) < 500
        except Exception:
            ok = False
        with lock:
            counters["requests"] += 1
            counters["successes" if ok else "failures"] += 1


def latest_artifact(case_id: str, repetition: int) -> Path | None:
    base = ROOT / "runtime_validation" / "artifacts" / case_id / f"rep-{repetition}"
    dirs = sorted([p for p in base.glob("*") if p.is_dir()]) if base.exists() else []
    return dirs[-1] if dirs else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("case_id", choices=["RT-11", "RT-12"])
    ap.add_argument("--repetition", type=int, required=True)
    ap.add_argument("--compose-file", required=True)
    ap.add_argument("--duration-seconds", type=int, default=30)
    ap.add_argument("--concurrency", type=int, default=30)
    args = ap.parse_args()

    case = load_case(args.case_id)
    if not 1 <= args.repetition <= int(case.get("repetitions", 1)):
        raise SystemExit("Repetition outside frozen manifest range")

    compose = args.compose_file
    params = case.get("parameters", {}) or {}
    concurrency = min(args.concurrency, int(params.get("concurrency", args.concurrency)))
    duration = min(args.duration_seconds, int(params.get("duration_seconds", args.duration_seconds)))

    run(["docker", "compose", "-f", compose, "up", "-d", "--scale", "front-end=1"])
    time.sleep(5)
    if args.case_id == "RT-12":
        replicas = int(params.get("replicas", 8))
        run(["docker", "compose", "-f", compose, "up", "-d", "--scale", f"front-end={replicas}"])
        time.sleep(5)

    url = resolve_frontend_url(compose)
    stop = threading.Event()
    counters = {"requests": 0, "successes": 0, "failures": 0}
    lock = threading.Lock()
    threads = [threading.Thread(target=worker, args=(url, stop, counters, lock), daemon=True) for _ in range(concurrency)]

    started = utc_now()
    for thread in threads:
        thread.start()
    time.sleep(max(3, min(8, duration // 4)))

    collector = ROOT / "runtime_validation" / "collect_runtime_artifacts.sh"
    run(["bash", str(collector), args.case_id, str(args.repetition), compose])

    remaining = max(0, duration - max(3, min(8, duration // 4)))
    time.sleep(remaining)
    stop.set()
    for thread in threads:
        thread.join(timeout=3)
    finished = utc_now()

    artifact = latest_artifact(args.case_id, args.repetition)
    if artifact is None:
        raise SystemExit("Runtime artifact was not created")
    evidence = {
        "case_id": args.case_id,
        "repetition": args.repetition,
        "started_utc": started,
        "finished_utc": finished,
        "url": url,
        "concurrency": concurrency,
        "duration_seconds": duration,
        **counters,
    }
    (artifact / "load_observation.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")

    run(["docker", "compose", "-f", compose, "up", "-d", "--scale", "front-end=1"])
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
