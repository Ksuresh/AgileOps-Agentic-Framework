"""Study 9 testbed: Online Boutique frontend on kind, release composition, governance actions, outcome measurement.

All operations act on the live cluster. Nothing here reads a correct answer: the runner records the incident snapshot
(what a decider sees) and the measured outcome of every action, and correctness is computed later from the outcomes.
"""
from __future__ import annotations

import copy
import json
import math
import re
import subprocess
import threading
import time
from pathlib import Path

from runtime_validation.run_online_boutique_case import measure_http

URL = "http://127.0.0.1:8080/"
STATE = Path("/tmp/aaf_outcome_state.json")
BASE_CPU_REQ_M = 100
SECRET_RE = re.compile(r"(?i)(secret|passw|token|api[_-]?key)")
MEASURE_S = 45
_trivy_cache: dict[str, int] = {}


def sh(cmd: list[str], *, check: bool = True, timeout: int = 300, capture: bool = True) -> str:
    cp = subprocess.run(cmd, check=check, timeout=timeout, text=True, capture_output=capture)
    return cp.stdout if capture else ""


def kjson(*args: str) -> dict:
    return json.loads(sh(["kubectl", *args, "-o", "json"]))


def save_base() -> dict:
    """Record the frozen healthy frontend deployment (image, env, resources) once per cluster."""
    d = kjson("get", "deployment/frontend")
    for k in ("status",):
        d.pop(k, None)
    md = d["metadata"]
    d["metadata"] = {"name": md["name"], "namespace": md.get("namespace", "default"), "labels": md.get("labels", {})}
    d["spec"]["strategy"] = {"type": "Recreate"}  # a broken release must not hide behind the old pod
    c = d["spec"]["template"]["spec"]["containers"][0]
    c["resources"] = {"requests": {"cpu": f"{BASE_CPU_REQ_M}m", "memory": "64Mi"}, "limits": {"cpu": "200m", "memory": "128Mi"}}
    STATE.write_text(json.dumps({"deployment": d, "stable_image": c["image"]}))
    return d


def base() -> dict:
    return json.loads(STATE.read_text())


def template(mods: list[str], case: dict, images: dict[str, str]) -> dict:
    """Frontend pod template for a release made of `mods` (see cases.yaml)."""
    st = base()
    d = copy.deepcopy(st["deployment"])
    c = d["spec"]["template"]["spec"]["containers"][0]
    env = [e for e in c.get("env", []) if e["name"] not in {"AAF_RELEASE", "AAF_OB_TEST_POLICY_MARKER", "PORT"}]
    port_env = [e for e in c.get("env", []) if e["name"] == "PORT"]
    if "feature" in mods:
        env.append({"name": "AAF_RELEASE", "value": "candidate"})
    if "secret" in mods:
        env.append({"name": "AAF_OB_TEST_POLICY_MARKER", "value": "exposed_test_secret"})
    if "crash" in mods:
        env.append({"name": "PORT", "value": "invalid-port"})
    else:
        env.extend(port_env)
    c["env"] = env
    if "vuln" in mods:
        c["image"] = images[case["vuln"]]
    if "perf" in mods:
        cpu = case["perf_cpu"]
        c["resources"] = {"requests": {"cpu": cpu, "memory": "64Mi"}, "limits": {"cpu": cpu, "memory": "128Mi"}}
    if "mitigated" in mods:
        c["resources"] = {"requests": {"cpu": "200m", "memory": "64Mi"}, "limits": {"cpu": "500m", "memory": "256Mi"}}
    d["spec"]["template"]["metadata"].setdefault("annotations", {})["aaf/release"] = ",".join(sorted(mods)) or "stable"
    d["spec"]["template"]["metadata"]["annotations"]["aaf/applied"] = str(time.time())
    return d


def apply(dep: dict, replicas: int) -> bool:
    dep = copy.deepcopy(dep)
    dep["spec"]["replicas"] = int(replicas)
    subprocess.run(["kubectl", "replace", "-f", "-"], input=json.dumps(dep), text=True, check=True, capture_output=True)
    ok = subprocess.run(["kubectl", "rollout", "status", "deployment/frontend", "--timeout=150s"], capture_output=True, text=True).returncode == 0
    time.sleep(8 if ok else 20)
    return ok


def pods() -> list[dict]:
    return kjson("get", "pods", "-l", "app=frontend")["items"]


def top_cpu_m() -> list[float]:
    out = subprocess.run(["kubectl", "top", "pods", "-l", "app=frontend", "--no-headers"], capture_output=True, text=True).stdout
    vals = []
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[1].endswith("m"):
            vals.append(float(parts[1][:-1]))
        elif len(parts) >= 2 and parts[1].isdigit():
            vals.append(float(parts[1]) * 1000)
    return vals


def critical_cves(image: str) -> int:
    if image not in _trivy_cache:
        out = sh(["trivy", "image", "--quiet", "--severity", "CRITICAL", "--format", "json", image], timeout=900)
        doc = json.loads(out)
        ids = {v["VulnerabilityID"] for r in doc.get("Results", []) or [] for v in r.get("Vulnerabilities", []) or []}
        _trivy_cache[image] = len(ids)
    return _trivy_cache[image]


def measure(load: int) -> dict:
    """Measure the current state: HTTP under fixed load, CPU during load, restarts, image, config, footprint."""
    samples: list[list[float]] = []

    def sampler():
        time.sleep(MEASURE_S * 0.5)
        for _ in range(3):
            samples.append(top_cpu_m())
            time.sleep(5)

    t = threading.Thread(target=sampler, daemon=True)
    t.start()
    http = measure_http(URL, MEASURE_S, load)
    t.join(timeout=30)
    dep = kjson("get", "deployment/frontend")
    c = dep["spec"]["template"]["spec"]["containers"][0]
    replicas = int(dep["spec"]["replicas"])
    req = c["resources"]["requests"]["cpu"]
    req_m = float(req[:-1]) if req.endswith("m") else float(req) * 1000
    lim = c["resources"]["limits"]["cpu"]
    lim_m = float(lim[:-1]) if lim.endswith("m") else float(lim) * 1000
    per_pod = [sum(s) / max(1, len(s)) for s in samples if s]
    cpu_mean = (sum(per_pod) / len(per_pod)) if per_pod else None
    restarts = sum(cs.get("restartCount", 0) for p in pods() for cs in (p.get("status", {}).get("containerStatuses") or []))
    env = {e["name"]: e.get("value", "") for e in c.get("env", [])}
    secret = any(SECRET_RE.search(str(v)) for v in env.values())
    stable_cves = critical_cves(base()["stable_image"])
    cves = critical_cves(c["image"])
    return {
        "http": http, "replicas": replicas, "cpu_request_m": req_m, "cpu_limit_m": lim_m, "footprint_cpu_m": replicas * req_m,
        "cpu_used_per_pod_m": cpu_mean, "utilisation_of_request": (cpu_mean / req_m) if cpu_mean is not None else None,
        "saturation_pct": (100.0 * cpu_mean / lim_m) if cpu_mean is not None else None,
        "restarts": restarts, "image": c["image"], "critical_cves_total": cves, "critical_cves_new": max(0, cves - stable_cves),
        "secret_in_config": secret, "feature_delivered": env.get("AAF_RELEASE") == "candidate",
        "release": dep["spec"]["template"]["metadata"].get("annotations", {}).get("aaf/release"),
    }


def scale_target(snapshot: dict) -> int:
    """Scale adjustment: HPA rule (target 60% of request), bounded to 1..6 replicas, from the incident measurement."""
    u = snapshot.get("utilisation_of_request")
    r = int(snapshot["replicas"])
    if u is None:
        return r
    return max(1, min(6, math.ceil(r * u / 0.6)))


ACTION_STATES = ("none", "rollback", "patch", "mitigate", "scale")


def action_release(action: str, case: dict, snapshot: dict) -> tuple[list[str], int]:
    """Release mods and replicas after applying a state-changing action to the incident state."""
    cur, prev, r = list(case["current"]), list(case["previous"]), int(case["replicas"])
    if action == "none":
        return cur, r
    if action == "rollback":
        return prev, r
    if action == "patch":
        return [m for m in cur if m not in {"vuln", "secret"}], r
    if action == "mitigate":
        return [m for m in cur if m != "perf"] + ["mitigated"], r
    if action == "scale":
        return cur, scale_target(snapshot)
    raise ValueError(action)
