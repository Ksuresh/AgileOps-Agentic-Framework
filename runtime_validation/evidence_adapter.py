from __future__ import annotations

"""Transform preserved runtime artifacts into auditable AAF evidence.

The adapter never reads intervention/oracle labels. Unknown values remain
explicitly missing and proxy values are labelled as proxies.
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return ""


def _read_glob(root: Path, pattern: str) -> str:
    return "\n".join(_read(p) for p in sorted(root.glob(pattern)))


def _pct(value: str) -> Optional[float]:
    m = re.search(r"(-?\d+(?:\.\d+)?)%", value or "")
    return float(m.group(1)) if m else None


def _parse_stats(text: str) -> Dict[str, Any]:
    rows = []
    for line in text.splitlines():
        if not line.strip() or line.lower().startswith("container"):
            continue
        parts = re.split(r"\s{2,}", line.strip())
        if len(parts) < 4:
            continue
        cpu = _pct(parts[2]) if len(parts) > 2 else None
        mem = _pct(parts[4]) if len(parts) > 4 else None
        rows.append({"raw": line, "cpu_pct": cpu, "memory_pct": mem})
    cpus = [r["cpu_pct"] for r in rows if r["cpu_pct"] is not None]
    mems = [r["memory_pct"] for r in rows if r["memory_pct"] is not None]
    return {
        "container_rows": len(rows),
        "max_cpu_pct": max(cpus) if cpus else None,
        "max_memory_pct": max(mems) if mems else None,
    }


def _normalise_service_name(name: str) -> str:
    value = str(name or "").strip().lstrip("/")
    value = re.sub(r"^(?:docker-compose|sock-shop)-", "", value)
    value = re.sub(r"-\d+$", "", value)
    return value


def _parse_ps(text: str) -> Dict[str, Any]:
    observed = []
    exited = []
    restarting = []
    unhealthy = []

    for line in text.splitlines():
        raw = line.strip()
        if not raw or raw.startswith("time=") or " level=warning " in raw:
            continue
        if raw.upper().startswith("NAME ") or raw.upper().startswith("CONTAINER "):
            continue
        parts = re.split(r"\s{2,}", raw)
        if len(parts) < 2:
            continue
        if len(parts) >= 6:
            service = _normalise_service_name(parts[3])
            status = parts[5]
        else:
            service = _normalise_service_name(parts[0])
            status = parts[-1]

        status_lower = status.lower()
        row = {"service": service, "status": status, "raw": line}
        observed.append(row)
        if "exited" in status_lower:
            m = re.search(r"exited\s*\((-?\d+)\)", status_lower)
            row["exit_code"] = int(m.group(1)) if m else None
            exited.append(row)
        if "restarting" in status_lower:
            restarting.append(row)
        if "unhealthy" in status_lower:
            unhealthy.append(row)

    return {
        "observed_container_rows": len(observed),
        "unhealthy_rows": len(unhealthy),
        "exited_rows_raw": len(exited),
        "restarting_rows": len(restarting),
        "exited": exited,
    }


def _compose_restart_policies(text: str) -> Dict[str, str]:
    if not text.strip():
        return {}
    try:
        payload = yaml.safe_load(text) or {}
    except yaml.YAMLError:
        return {}
    services = payload.get("services", {}) if isinstance(payload, dict) else {}
    if not isinstance(services, dict):
        return {}
    policies: Dict[str, str] = {}
    for name, spec in services.items():
        if isinstance(spec, dict):
            policies[_normalise_service_name(str(name))] = str(spec.get("restart", "") or "").strip().lower()
    return policies


def _inspect_process_states(root: Path) -> Dict[str, Any]:
    unexpected_exited = 0
    expected_completed = 0
    restarting = 0
    inspected = 0
    restart_counts = []
    restart_policy_by_service: Dict[str, str] = {}

    for path in sorted(root.glob("inspect/*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        except (json.JSONDecodeError, OSError):
            continue
        items = payload if isinstance(payload, list) else [payload]
        for item in items:
            if not isinstance(item, dict):
                continue
            inspected += 1
            state = item.get("State", {}) or {}
            host = item.get("HostConfig", {}) or {}
            config = item.get("Config", {}) or {}
            labels = config.get("Labels", {}) or {}
            service = _normalise_service_name(labels.get("com.docker.compose.service") or item.get("Name") or path.stem)
            restart_policy = str((host.get("RestartPolicy", {}) or {}).get("Name", "") or "").lower()
            restart_policy_by_service[service] = restart_policy
            status = str(state.get("Status", "")).lower()
            exit_code = state.get("ExitCode")
            oom = bool(state.get("OOMKilled", False))
            try:
                restart_counts.append(int(item.get("RestartCount", 0) or 0))
            except (TypeError, ValueError):
                pass
            if status == "restarting":
                restarting += 1
            elif status == "exited":
                if exit_code == 0 and not oom and restart_policy in {"", "no"}:
                    expected_completed += 1
                else:
                    unexpected_exited += 1

    return {
        "inspected_containers": inspected,
        "unexpected_exited": unexpected_exited,
        "expected_completed": expected_completed,
        "restarting": restarting,
        "restart_count_max": max(restart_counts) if restart_counts else None,
        "restart_policy_by_service": restart_policy_by_service,
    }


def _classify_exited_rows(ps: Dict[str, Any], compose_policies: Dict[str, str], inspect_policies: Dict[str, str]) -> Dict[str, int]:
    continuous_policies = {"always", "unless-stopped", "on-failure"}
    unexpected = 0
    expected = 0
    for row in ps.get("exited", []):
        service = _normalise_service_name(row.get("service", ""))
        policy = compose_policies.get(service, inspect_policies.get(service, ""))
        exit_code = row.get("exit_code")
        if policy in continuous_policies or (exit_code is not None and int(exit_code) != 0):
            unexpected += 1
        else:
            expected += 1
    return {"unexpected": unexpected, "expected": expected}


def derive_observables(case_dir: str | Path) -> Dict[str, Any]:
    p = Path(case_dir)
    ps = _parse_ps(_read(p / "compose_ps.txt") or _read(p / "docker_ps.txt"))
    states = _inspect_process_states(p)
    stats = _parse_stats(_read(p / "docker_stats.txt"))
    logs = _read_glob(p, "logs/*.log")
    config = _read(p / "compose_resolved.yaml")
    compose_policies = _compose_restart_policies(config)
    classified = _classify_exited_rows(ps, compose_policies, states["restart_policy_by_service"])

    if ps["exited_rows_raw"]:
        exited_rows = classified["unexpected"]
        expected_completed_rows = classified["expected"]
    else:
        exited_rows = states["unexpected_exited"]
        expected_completed_rows = states["expected_completed"]
    restarting_rows = max(ps["restarting_rows"], states["restarting"])

    return {
        "availability_proxy": {
            "unhealthy_rows": ps["unhealthy_rows"],
            "exited_rows": exited_rows,
            "expected_completed_rows": expected_completed_rows,
            "restarting_rows": restarting_rows,
            "source": "docker compose process state + resolved restart policy + docker inspect",
        },
        "restart_count_max": states["restart_count_max"],
        "resource_observation": stats,
        "container_footprint": ps["observed_container_rows"],
        "error_log_line_count": sum(1 for ln in logs.splitlines() if re.search(r"\b(error|fatal|panic|exception)\b", ln, re.I)),
        "artifact_presence": {
            "logs": bool(logs.strip()),
            "inspect": states["inspected_containers"] > 0,
            "compose_config": bool(config.strip()),
            "stats": bool(_read(p / "docker_stats.txt").strip()),
        },
    }


def _ev(status: str, source: str, note: str = "") -> Dict[str, str]:
    return {"status": status, "source": source, "note": note}


def to_aaf_telemetry(observables: Dict[str, Any], baseline: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    av = observables.get("availability_proxy", {})
    restart = observables.get("restart_count_max")
    resource = observables.get("resource_observation", {})
    footprint = observables.get("container_footprint")
    service_problem = any((av.get("unhealthy_rows", 0), av.get("exited_rows", 0), av.get("restarting_rows", 0)))

    deploy = {
        "restart_loops": restart,
        "_evidence": {
            "restart_loops": _ev("measured" if restart is not None else "missing", "docker inspect RestartCount"),
            "pipeline_failed": _ev("missing", "CI/CD evidence not collected"),
            "config_drift": _ev("missing", "configuration-diff evidence not yet classified"),
            "rollback_marker": _ev("missing", "release metadata not collected"),
            "artifact_mismatch": _ev("missing", "release/image expectation evidence not collected"),
        },
    }
    sre = {
        "saturation_pct": resource.get("max_cpu_pct"),
        "availability_pct": 0.0 if service_problem else 99.9,
        "_evidence": {
            "p95_latency_ms": _ev("missing", "request latency instrumentation not present"),
            "error_rate_pct": _ev("missing", "request error instrumentation not present"),
            "saturation_pct": _ev("proxy" if resource.get("max_cpu_pct") is not None else "missing", "docker stats max CPU percentage", "Container CPU proxy, not SLO saturation"),
            "availability_pct": _ev("proxy", "docker compose process state + resolved restart policy", "Binary process-state proxy; clean completion is excluded only for non-continuous services; not time-window SLO availability"),
        },
    }
    finops = {"_evidence": {
        "cost_spike_pct": _ev("missing", "no monetary billing source"),
        "hpa_scale_to": _ev("missing", "Docker Compose footprint is not Kubernetes HPA"),
        "cpu_request_increase_pct": _ev("missing", "resource request history not collected"),
        "memory_request_increase_pct": _ev("missing", "resource request history not collected"),
    }}
    if baseline and isinstance(footprint, (int, float)):
        base_fp = baseline.get("container_footprint")
        if isinstance(base_fp, (int, float)) and base_fp > 0:
            finops["cost_spike_pct"] = 100.0 * (footprint - base_fp) / base_fp
            finops["_evidence"]["cost_spike_pct"] = _ev("proxy", "container footprint relative to healthy baseline", "Resource-footprint proxy only; not monetary cost")
    sec = {"_evidence": {
        "critical_cves": _ev("missing", "security scanner evidence absent"),
        "policy_violation": _ev("missing", "policy-as-code evidence absent"),
        "iam_drift": _ev("missing", "IAM evidence absent"),
        "compliance_gap": _ev("missing", "compliance evidence absent"),
    }}
    return {
        "deploy": deploy,
        "sre": sre,
        "finops": finops,
        "sec": sec,
        "_runtime_observables": observables,
        "_evidence_schema_version": "2.2",
    }


def write_evidence(case_dir: str | Path, baseline_dir: Optional[str | Path] = None) -> Dict[str, Any]:
    p = Path(case_dir)
    observables = derive_observables(p)
    baseline = derive_observables(baseline_dir) if baseline_dir else None
    telemetry = to_aaf_telemetry(observables, baseline)
    (p / "derived_observables.json").write_text(json.dumps(observables, indent=2), encoding="utf-8")
    (p / "aaf_telemetry.json").write_text(json.dumps(telemetry, indent=2), encoding="utf-8")
    return telemetry
