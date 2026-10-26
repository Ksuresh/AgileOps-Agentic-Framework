"""Study 16 calibration on Weaveworks Sock Shop (no action outcomes are measured). Same fixed rules as Study 15
(outcome_study/calibrate15.py), applied to the front-end and catalogue services with Sock Shop's published image tags:

0. Healthy storefront: three 45 s measurements of GET /catalogue with every service at its stable release; the violation
   threshold is twice the median healthy p95 (the Study 9 rule), error threshold 1%.
1. Vulnerable images (front-end SF1, SF2; catalogue SC1, SC2): older tags, newest first; >= 1 critical CVE more than the
   stable image, rolls out and the storefront serves; first two.
2. Low-risk image (NSF, NSC): older tags, newest first; no more critical CVEs than stable and more HIGH/MEDIUM findings;
   first that qualifies, else the stable image is used.
3. Degradation level: CPU limit 50m, 30m, 20m, 10m, 5m, largest that violates the threshold (no latency setting exists
   in these services; if none violates, 5m is used and recorded).
4. Crash check: an entrypoint that does not exist (CRASH_MODE=command); rollout, restarts and storefront errors recorded."""
from __future__ import annotations

import json
import os
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from outcome_study import calibrate15 as C  # noqa: E402

C.REGISTRIES = ["docker.io/weaveworksdemos"]
C.LATENCY = []
C.LOAD = 20
TAGS = {"front-end": ["0.3.11", "0.3.10", "0.3.9", "0.3.8", "0.3.7", "0.3.6", "0.3.5", "0.3.4", "0.3.3", "0.3.1", "0.3.0"],
        "catalogue": ["0.3.4", "0.3.3", "0.3.2", "0.3.1", "0.3.0", "0.2.0"]}
C.KEYS = {"front-end": ("SF1", "SF2", "NSF"), "catalogue": ("SC1", "SC2", "NSC")}
C.USED = {"front-end": set(), "catalogue": set()}


def healthy() -> dict:
    os.environ["TARGET_SERVICE"] = "front-end"
    import importlib
    from outcome_study import testbed as T
    importlib.reload(T)
    T.save_base()
    ms = [T.measure(C.LOAD) for _ in range(3)]
    p95 = [m["http"]["latency_p95_ms"] for m in ms]
    return {"healthy_measurements": ms, "healthy_p95_ms": p95, "healthy_error_pct": [m["http"]["error_rate_pct"] for m in ms],
            "healthy_p95_median_ms": statistics.median(p95)}


def main() -> None:
    h = healthy()
    C.P95_MAX = 2 * h["healthy_p95_median_ms"]
    print("healthy", h["healthy_p95_ms"], h["healthy_error_pct"], "threshold", C.P95_MAX, flush=True)
    res = {}
    for svc in ("front-end", "catalogue"):
        C.VULN_TAGS = TAGS[svc]
        C.LOWRISK_TAGS = TAGS[svc]
        res[svc] = C.calibrate(svc)
    cal = {"application": "Weaveworks Sock Shop", **h, "p95_threshold_ms": C.P95_MAX,
           "stable_images": {s: r["stable"] for s, r in res.items()},
           "vuln_images": {"A": None, "B": None}, "perf_cpu": {s: r["perf_cpu"] for s, r in res.items()}, "study16_calibration": res}
    for r in res.values():
        cal["vuln_images"].update(r["images"])
    cal["vuln_images"]["A"] = cal["vuln_images"]["SF1"] or "unused"  # run_outcomes.py checks that A and B exist
    cal["vuln_images"]["B"] = cal["vuln_images"]["SF2"] or "unused"
    out = ROOT / "outcome_study16" / "calibration"
    out.mkdir(parents=True, exist_ok=True)
    (out / "calibration.json").write_text(json.dumps(cal, indent=2))
    print(json.dumps({"images": cal["vuln_images"], "perf_cpu": cal["perf_cpu"], "threshold": C.P95_MAX}, indent=2))


if __name__ == "__main__":
    main()
