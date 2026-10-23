"""Study 9 calibration (healthy states only; no action outcomes are measured here).

1. Healthy stable frontend, 1 replica, normal load: three 45 s measurements -> healthy p95 and error rate.
2. Vulnerable-image selection rule (PROTOCOL.md): candidate Online Boutique frontend tags, newest first; keep tags that pull,
   serve HTTP 200 with the stable configuration, and have >= 1 critical CVE more than the stable image (trivy).
   A = first such tag, B = second.
"""
from __future__ import annotations

import json
import statistics
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from outcome_study import testbed as T  # noqa: E402

CANDIDATES = [f"{reg}/google-samples/microservices-demo/frontend:{tag}"
              for tag in ("v0.8.0", "v0.7.0", "v0.6.0", "v0.5.0", "v0.4.0", "v0.3.0", "v0.2.0", "v0.1.0")
              for reg in ("us-central1-docker.pkg.dev", "gcr.io")]


def serves() -> bool:
    ok = 0
    for _ in range(5):
        try:
            with urllib.request.urlopen(T.URL, timeout=3) as r:
                ok += r.status == 200
        except Exception:
            pass
    return ok >= 4


def main() -> None:
    out = ROOT / "outcome_study" / "calibration"
    out.mkdir(parents=True, exist_ok=True)
    T.save_base()
    healthy = []
    for _ in range(3):
        T.apply(T.template([], {}, {}), 1)
        healthy.append(T.measure(20))
    p95 = [h["http"]["latency_p95_ms"] for h in healthy]
    err = [h["http"]["error_rate_pct"] for h in healthy]
    stable = T.base()["stable_image"]
    scans = []
    chosen = []
    for img in CANDIDATES:
        rec = {"image": img}
        try:
            rec["critical_cves"] = T.critical_cves(img)
        except Exception as exc:
            rec["error"] = f"scan: {str(exc)[:200]}"
            scans.append(rec)
            continue
        rec["critical_cves_new"] = max(0, rec["critical_cves"] - T.critical_cves(stable))
        if rec["critical_cves_new"] >= 1 and len(chosen) < 2:
            rec["rollout_ok"] = T.apply(T.template(["vuln"], {"vuln": "X"}, {"X": img}), 1)
            rec["serves_http_200"] = serves()
            if rec["rollout_ok"] and rec["serves_http_200"] and not any(c.split(":")[-1] == img.split(":")[-1] for c in chosen):
                chosen.append(img)
        scans.append(rec)
    T.apply(T.template([], {}, {}), 1)
    doc = {"stable_image": stable, "stable_critical_cves": T.critical_cves(stable), "healthy_measurements": healthy,
           "healthy_p95_ms": p95, "healthy_error_pct": err, "healthy_p95_median_ms": statistics.median(p95),
           "image_scans": scans, "vuln_images": {"A": chosen[0] if chosen else None, "B": chosen[1] if len(chosen) > 1 else None}}
    (out / "calibration.json").write_text(json.dumps(doc, indent=2))
    print(json.dumps({k: doc[k] for k in ("stable_image", "stable_critical_cves", "healthy_p95_ms", "healthy_error_pct", "vuln_images")}, indent=2))


if __name__ == "__main__":
    main()
