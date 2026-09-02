from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = Path(__file__).with_name("interventions.yaml")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run(cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess:
    print("+", " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=ROOT, text=True, check=check)


def load_case(case_id: str) -> dict:
    data = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    for case in data["cases"]:
        if case["id"] == case_id:
            return case
    raise SystemExit(f"Unknown case: {case_id}")


def write_metadata(case: dict, repetition: int, phase: str, extra: dict | None = None) -> Path:
    out = ROOT / "runtime_validation" / "run_metadata" / case["id"] / f"rep-{repetition}"
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "case_id": case["id"],
        "title": case["title"],
        "repetition": repetition,
        "phase": phase,
        "timestamp_utc": utc_now(),
        "intervention": case.get("intervention"),
        "target_service": case.get("target_service"),
        "parameters": case.get("parameters", {}),
        # Oracle is recorded in experiment metadata but must never be supplied
        # to the AAF inference path.
        "oracle_domains": case.get("oracle_domains", []),
        "admissible_actions": case.get("admissible_actions", []),
    }
    if extra:
        payload.update(extra)
    path = out / f"{phase}.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def apply_supported_intervention(case: dict, compose_file: str) -> None:
    kind = case["intervention"]
    service = case.get("target_service")
    p = case.get("parameters", {})

    if kind == "none":
        return
    if kind == "stop_service":
        run(["docker", "compose", "-f", compose_file, "stop", service])
        return
    if kind == "restart_loop":
        for _ in range(int(p.get("count", 5))):
            run(["docker", "compose", "-f", compose_file, "restart", service])
            time.sleep(float(p.get("interval_seconds", 5)))
        return
    if kind == "scale_service":
        run(["docker", "compose", "-f", compose_file, "up", "-d", "--scale", f"{service}={int(p['replicas'])}"])
        return

    # Deliberately fail closed for interventions that require an explicit,
    # benchmark-specific implementation. This prevents the experiment runner
    # from silently fabricating a fault or proxy.
    raise SystemExit(
        f"Intervention '{kind}' is specified but not yet automated. "
        "Implement and review it before collecting this case."
    )


def restore(compose_file: str) -> None:
    run(["docker", "compose", "-f", compose_file, "down"], check=False)
    run(["docker", "compose", "-f", compose_file, "up", "-d"])


def collect(case_id: str, repetition: int, compose_file: str) -> None:
    collector = ROOT / "runtime_validation" / "collect_runtime_artifacts.sh"
    run(["bash", str(collector), case_id, str(repetition), compose_file])


def main() -> None:
    ap = argparse.ArgumentParser(description="Run one controlled Sock Shop runtime-validation case.")
    ap.add_argument("case_id")
    ap.add_argument("--repetition", type=int, default=1)
    ap.add_argument("--compose-file", default="docker-compose.yml")
    ap.add_argument("--settle-seconds", type=float, default=20.0)
    ap.add_argument("--restore-first", action="store_true")
    ap.add_argument("--restore-after", action="store_true")
    args = ap.parse_args()

    case = load_case(args.case_id)
    if args.repetition < 1 or args.repetition > int(case.get("repetitions", 1)):
        raise SystemExit("Repetition outside manifest range")

    if args.restore_first:
        restore(args.compose_file)
        time.sleep(args.settle_seconds)

    write_metadata(case, args.repetition, "before_intervention")
    apply_supported_intervention(case, args.compose_file)
    write_metadata(case, args.repetition, "after_intervention")
    time.sleep(args.settle_seconds)
    collect(case["id"], args.repetition, args.compose_file)
    write_metadata(case, args.repetition, "after_collection")

    if args.restore_after:
        restore(args.compose_file)
        write_metadata(case, args.repetition, "after_restore")


if __name__ == "__main__":
    main()
