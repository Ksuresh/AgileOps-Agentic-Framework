"""Study 15 calibration (no action outcomes are measured). Fixed selection rules, per service, applied before the case
file is completed (PROTOCOL.md, outcome_study15):

1. Vulnerable images: candidate tags newest first, gcr.io then us-central1-docker.pkg.dev; keep images that scan, carry
   >= 1 critical CVE more than the service's stable image, roll out and keep the storefront serving HTTP 200. The first two
   qualifying tags not used in Studies 9-14 are chosen (frontend G, H; productcatalogservice P3, P4; currencyservice C1, C2);
   only if fewer than two qualify are previously used tags considered, in the same order.
2. Low-risk images: tags newer than the vulnerable range, newest first; keep images with no more critical CVEs than the
   stable image and more HIGH/MEDIUM findings than it, that roll out and serve. The first one is chosen (NF, NP, NC). If no
   tag qualifies, the low-risk cases of that service use the stable image (as in Study 14).
3. Degradation level: CPU limit candidates 50m, 30m, 20m, 10m, 5m, largest first, applied to the service at 1 replica under
   normal load; the largest that violates the Study 9 reliability threshold (p95 > 816 ms or errors > 1%) is the service's
   perf level. Amendment 1: if none violates, injected latency through the service's EXTRA_LATENCY setting, candidates
   100ms to 1600ms, mildest first; the first that violates (else 1600ms, recorded).
4. Crash check (descriptive): the invalid-port release is applied once per service and its rollout and storefront errors
   are recorded.
The healthy storefront latency with each service at its stable release is recorded for reference."""
from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from outcome_study.calibrate import serves  # noqa: E402

VULN_TAGS = ["v0.9.0", "v0.8.0", "v0.7.0", "v0.6.0", "v0.5.0", "v0.4.0", "v0.3.0", "v0.2.0", "v0.1.0"]
LOWRISK_TAGS = ["v0.10.5", "v0.10.4", "v0.10.3", "v0.10.2", "v0.10.1", "v0.10.0", "v0.9.0"]
REGISTRIES = ["gcr.io/google-samples/microservices-demo", "us-central1-docker.pkg.dev/google-samples/microservices-demo"]
USED = {"frontend": {"v0.8.0", "v0.7.0", "v0.6.0", "v0.5.0", "v0.4.0", "v0.3.0"},
        "productcatalogservice": {"v0.8.0", "v0.7.0"}, "currencyservice": set()}
KEYS = {"frontend": ("G", "H", "NF"), "productcatalogservice": ("P3", "P4", "NP"), "currencyservice": ("C1", "C2", "NC")}
PERF = ["50m", "30m", "20m", "10m", "5m"]
LATENCY = ["100ms", "200ms", "400ms", "800ms", "1600ms"]  # Amendment 1, used only if no CPU level violates
P95_MAX, ERR_MAX = 816.0, 1.0
LOAD = 20  # normal load (cases.yaml)


def scan(T, img: str) -> dict:
    rec = {"image": img}
    try:
        rec["critical_cves"] = T.critical_cves(img)
        rec["noncritical_cves"] = T.noncritical_cves(img)
    except Exception as exc:  # tag absent from this registry, or scan failure
        rec["error"] = str(exc)[:200]
    return rec


def rolls_and_serves(T, img: str, rec: dict) -> bool:
    rec["rollout_ok"] = T.apply(T.template(["vuln"], {"vuln": "X"}, {"X": img}), 1)
    rec["serves_http_200"] = serves()
    T.apply(T.template([], {}, {}), 1)
    return rec["rollout_ok"] and rec["serves_http_200"]


def calibrate(service: str) -> dict:
    os.environ["TARGET_SERVICE"] = service
    from outcome_study import testbed as T
    importlib.reload(T)
    T.save_base()
    stable = T.base()["stable_image"]
    st = {"image": stable, "critical_cves": T.critical_cves(stable), "noncritical_cves": T.noncritical_cves(stable)}
    out = {"stable": st, "vuln_scans": [], "lowrisk_scans": [], "perf_scans": [], "images": {}}
    healthy = T.measure(LOAD)
    out["healthy"] = {"p95_ms": healthy["http"]["latency_p95_ms"], "error_rate_pct": healthy["http"]["error_rate_pct"]}
    k1, k2, kn = KEYS[service]
    chosen: list[str] = []
    # first pass: tags not used in Studies 9-14; second pass (only if fewer than two qualified): previously used tags
    for tag in [t for t in VULN_TAGS if t not in USED[service]] + [t for t in VULN_TAGS if t in USED[service]]:
        if len(chosen) >= 2:
            break
        for reg in REGISTRIES:
            rec = scan(T, f"{reg}/{service}:{tag}")
            out["vuln_scans"].append(rec)
            if "error" in rec:
                continue
            if rec["critical_cves"] > st["critical_cves"] and rolls_and_serves(T, rec["image"], rec):
                chosen.append(rec["image"])
            break  # first registry that scans decides for this tag
    out["images"][k1] = chosen[0] if chosen else None
    out["images"][k2] = chosen[1] if len(chosen) > 1 else None
    out["images"][kn] = None
    for tag in LOWRISK_TAGS:
        if out["images"][kn]:
            break
        for reg in REGISTRIES:
            rec = scan(T, f"{reg}/{service}:{tag}")
            out["lowrisk_scans"].append(rec)
            if "error" in rec:
                continue
            if (rec["critical_cves"] <= st["critical_cves"] and rec["noncritical_cves"] > st["noncritical_cves"]
                    and rolls_and_serves(T, rec["image"], rec)):
                out["images"][kn] = rec["image"]
            break
    out["perf_cpu"] = None
    for cpu in PERF:
        T.apply(T.template(["perf"], {"perf_cpu": cpu}, {}), 1)
        m = T.measure(LOAD)
        rec = {"cpu": cpu, "p95_ms": m["http"]["latency_p95_ms"], "error_rate_pct": m["http"]["error_rate_pct"]}
        rec["violates"] = rec["p95_ms"] > P95_MAX or rec["error_rate_pct"] > ERR_MAX
        out["perf_scans"].append(rec)
        print(service, "perf", rec, flush=True)
        if rec["violates"]:
            out["perf_cpu"] = cpu
            break
    if out["perf_cpu"] is None:  # Amendment 1: injected latency for services that CPU starvation does not degrade
        for lat in LATENCY:
            T.apply(T.template(["perf"], {"perf_cpu": "lat:" + lat}, {}), 1)
            m = T.measure(LOAD)
            rec = {"latency": lat, "p95_ms": m["http"]["latency_p95_ms"], "error_rate_pct": m["http"]["error_rate_pct"]}
            rec["violates"] = rec["p95_ms"] > P95_MAX or rec["error_rate_pct"] > ERR_MAX
            out["perf_scans"].append(rec)
            print(service, "perf", rec, flush=True)
            if rec["violates"]:
                out["perf_cpu"] = "lat:" + lat
                break
    if out["perf_cpu"] is None:
        out["perf_cpu"], out["perf_note"] = "lat:" + LATENCY[-1], "no candidate violated; strongest used"
    ok = T.apply(T.template(["crash"], {}, {}), 1)
    m = T.measure(LOAD)
    out["crash_check"] = {"rollout_ok": ok, "storefront_error_rate_pct": m["http"]["error_rate_pct"], "restarts": m["restarts"]}
    T.apply(T.template([], {}, {}), 1)
    print(service, json.dumps(out["images"]), out["perf_cpu"], out["crash_check"], flush=True)
    return out


def main() -> None:
    base = json.loads((ROOT / "outcome_study14" / "calibration" / "calibration.json").read_text())
    base.pop("study14_scans", None)
    res = {svc: calibrate(svc) for svc in ("frontend", "productcatalogservice", "currencyservice")}
    for svc, r in res.items():
        base["vuln_images"].update({k: v for k, v in r["images"].items()})
    base["perf_cpu"] = {svc: r["perf_cpu"] for svc, r in res.items()}
    base["study15_calibration"] = res
    base["note"] = ("Study 15: Study 9 thresholds; images G, H, NF (frontend), P3, P4, NP (productcatalogservice), "
                    "C1, C2, NC (currencyservice) and per-service perf levels selected by calibrate15.py.")
    out = ROOT / "outcome_study15" / "calibration"
    out.mkdir(parents=True, exist_ok=True)
    (out / "calibration.json").write_text(json.dumps(base, indent=2))
    print(json.dumps({"images": base["vuln_images"], "perf_cpu": base["perf_cpu"]}, indent=2))


if __name__ == "__main__":
    main()
