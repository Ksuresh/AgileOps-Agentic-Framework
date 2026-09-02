from __future__ import annotations

"""Transform preserved runtime artifacts into auditable AAF evidence.

Scientific constraint: this adapter must never read intervention oracle labels.
It only reads runtime artifacts/observables. Unknown values remain missing; the
adapter does not invent latency, error-rate, cost, CVE, or policy measurements.
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, Optional


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return ""


def _pct(value: str) -> Optional[float]:
    m = re.search(r"(-?\d+(?:\.\d+)?)%", value or "")
    return float(m.group(1)) if m else None


def _parse_stats(text: str) -> Dict[str, Any]:
    """Best-effort parser for docker stats --no-stream table output."""
    rows = []
    for line in text.splitlines():
        if not line.strip() or line.lower().startswith("container"):
            continue
        parts = re.split(r"\s{2,}", line.strip())
        if len(parts) < 4:
            continue
        cpu = _pct(parts[2]) if len(parts) > 2 else None
        mem = _pct(parts[6]) if len(parts) > 6 else None
        rows.append({"raw": line, "cpu_pct": cpu, "memory_pct": mem})
    cpus = [r["cpu_pct"] for r in rows if r["cpu_pct"] is not None]
    mems = [r["memory_pct"] for r in rows if r["memory_pct"] is not None]
    return {
        "container_rows": len(rows),
        "max_cpu_pct": max(cpus) if cpus else None,
        "max_memory_pct": max(mems) if mems else None,
    }


def _parse_ps(text: str) -> Dict[str, Any]:
    lower = text.lower()
    unhealthy = len(re.findall(r"\bunhealthy\b", lower))
    exited = len(re.findall(r"\bexited\b", lower))
    restarting = len(re.findall(r"\brestarting\b", lower))
    # Count table rows conservatively; this is a footprint observable, not cost.
    rows = [ln for ln in text.splitlines() if ln.strip()]
    if rows and ("name" in rows[0].lower() or "container" in rows[0].lower()):
        rows = rows[1:]
    return {
        "observed_container_rows": len(rows),
        "unhealthy_rows": unhealthy,
        "exited_rows": exited,
        "restarting_rows": restarting,
    }


def derive_observables(case_dir: str | Path) -> Dict[str, Any]:
    p = Path(case_dir)
    ps = _parse_ps(_read(p / "docker_ps.txt") or _read(p / "compose_ps.txt"))
    stats = _parse_stats(_read(p / "docker_stats.txt"))
    logs = _read(p / "service_logs.txt")
    inspect = _read(p / "container_inspect.json")
    config = _read(p / "compose_config.txt")

    restart_counts = [int(x) for x in re.findall(r'"RestartCount"\s*:\s*(\d+)', inspect)]
    error_lines = sum(1 for ln in logs.splitlines() if re.search(r"\b(error|fatal|panic|exception)\b", ln, re.I))

    return {
        "availability_proxy": {
            "unhealthy_rows": ps["unhealthy_rows"],
            "exited_rows": ps["exited_rows"],
            "restarting_rows": ps["restarting_rows"],
            "source": "docker/compose process state",
        },
        "restart_count_max": max(restart_counts) if restart_counts else None,
        "resource_observation": stats,
        "container_footprint": ps["observed_container_rows"],
        "error_log_line_count": error_lines,
        "artifact_presence": {
            "logs": bool(logs.strip()),
            "inspect": bool(inspect.strip()),
            "compose_config": bool(config.strip()),
            "stats": bool(_read(p / "docker_stats.txt").strip()),
        },
    }


def to_aaf_telemetry(observables: Dict[str, Any], baseline: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Map only defensible observables into the current AAF schema.

    Fields without a direct measurement are marked missing rather than imputed.
    This intentionally exposes schema gaps that should be fixed before the
    runtime study is reported.
    """
    av = observables.get("availability_proxy", {})
    restart = observables.get("restart_count_max")
    resource = observables.get("resource_observation", {})
    footprint = observables.get("container_footprint")

    service_problem = any((av.get("unhealthy_rows", 0), av.get("exited_rows", 0), av.get("restarting_rows", 0)))

    deploy: Dict[str, Any] = {
        "pipeline_failed": False,
        "config_drift": False,
        "rollback_marker": False,
        "artifact_mismatch": False,
        "restart_loops": restart or 0,
        "_provenance": {"restart_loops": "container inspect RestartCount" if restart is not None else "unobserved"},
    }

    # Current SRE agent requires latency/error/saturation/availability numbers.
    # Docker state alone cannot justify latency/error percentages. Only resource
    # saturation is populated when docker stats supplies a percentage.
    sre: Dict[str, Any] = {
        "p95_latency_ms": 0.0,
        "error_rate_pct": 0.0,
        "saturation_pct": resource.get("max_cpu_pct") or 0.0,
        "availability_pct": 0.0 if service_problem else 99.9,
        "_provenance": {
            "p95_latency_ms": "unobserved; neutral placeholder required by legacy schema",
            "error_rate_pct": "unobserved; neutral placeholder required by legacy schema",
            "saturation_pct": "max docker CPU percentage proxy" if resource.get("max_cpu_pct") is not None else "unobserved",
            "availability_pct": "binary process-state proxy; not measured SLO availability",
        },
    }

    finops: Dict[str, Any] = {
        "cost_spike_pct": 0.0,
        "hpa_scale_to": 0,
        "cpu_request_increase_pct": 0.0,
        "memory_request_increase_pct": 0.0,
        "_provenance": {"cost_spike_pct": "unobserved; no cloud billing data"},
    }
    if baseline and footprint is not None:
        base_fp = baseline.get("container_footprint")
        if isinstance(base_fp, (int, float)) and base_fp > 0 and footprint > base_fp:
            # This is explicitly a footprint proxy, not measured cost.
            finops["cost_spike_pct"] = 100.0 * (footprint - base_fp) / base_fp
            finops["_provenance"]["cost_spike_pct"] = "container-footprint increase proxy; not monetary cost"

    sec: Dict[str, Any] = {
        "critical_cves": 0,
        "policy_violation": False,
        "iam_drift": False,
        "compliance_gap": False,
        "_missing": True,
        "_provenance": {"status": "security scanner/policy evidence not present in generic Docker artifacts"},
    }

    return {"deploy": deploy, "sre": sre, "finops": finops, "sec": sec, "_runtime_observables": observables}


def write_evidence(case_dir: str | Path, baseline_dir: Optional[str | Path] = None) -> Dict[str, Any]:
    p = Path(case_dir)
    observables = derive_observables(p)
    baseline = derive_observables(baseline_dir) if baseline_dir else None
    telemetry = to_aaf_telemetry(observables, baseline)
    (p / "derived_observables.json").write_text(json.dumps(observables, indent=2), encoding="utf-8")
    (p / "aaf_telemetry.json").write_text(json.dumps(telemetry, indent=2), encoding="utf-8")
    return telemetry
