"""Study 14 calibration (no action outcomes measured). Fixed selection rule, per service, newest tag first:
keep images that scan, carry >= 1 critical CVE more than that service's stable image, roll out and keep the storefront
serving HTTP 200. frontend: E, F = first two qualifying tags not used in Studies 9-13 (A-D). productcatalogservice: P1, P2.
Also records each stable image's HIGH/MEDIUM finding count (low-risk releases use the stable image)."""
from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from outcome_study.calibrate import serves  # noqa: E402

TAGS = ["v0.8.0", "v0.7.0", "v0.6.0", "v0.5.0", "v0.4.0", "v0.3.0", "v0.2.0", "v0.1.0"]
USED = {"v0.8.0", "v0.7.0", "v0.6.0", "v0.5.0"}  # frontend images A-D (Studies 9, 12, 13)


def select(service: str, keys: tuple[str, str], skip: set[str]) -> tuple[dict, list]:
    os.environ["TARGET_SERVICE"] = service
    from outcome_study import testbed as T
    importlib.reload(T)
    T.save_base()
    stable = T.base()["stable_image"]
    st = {"image": stable, "critical_cves": T.critical_cves(stable), "noncritical_cves": T.noncritical_cves(stable)}
    chosen, scans = [], []
    for tag in TAGS:
        if tag in skip:
            continue
        img = f"gcr.io/google-samples/microservices-demo/{service}:{tag}"
        rec = {"image": img}
        try:
            rec["critical_cves"] = T.critical_cves(img)
        except Exception as exc:
            rec["error"] = str(exc)[:200]; scans.append(rec); continue
        if rec["critical_cves"] > st["critical_cves"] and len(chosen) < 2:
            rec["rollout_ok"] = T.apply(T.template(["vuln"], {"vuln": "X"}, {"X": img}), 1)
            rec["serves_http_200"] = serves()
            if rec["rollout_ok"] and rec["serves_http_200"]:
                chosen.append(img)
        scans.append(rec)
    T.apply(T.template([], {}, {}), 1)
    return {keys[0]: chosen[0] if chosen else None, keys[1]: chosen[1] if len(chosen) > 1 else None}, [st] + scans


def main() -> None:
    base = json.loads((ROOT / "outcome_study12" / "calibration" / "calibration.json").read_text())
    fe, fe_scans = select("frontend", ("E", "F"), USED)
    pc, pc_scans = select("productcatalogservice", ("P1", "P2"), set())
    base["vuln_images"].update(fe); base["vuln_images"].update(pc)
    base["study14_scans"] = {"frontend": fe_scans, "productcatalogservice": pc_scans}
    base["note"] = "Study 14: Study 9 thresholds; images E, F (frontend) and P1, P2 (productcatalogservice) selected by calibrate14.py."
    out = ROOT / "outcome_study14" / "calibration"
    out.mkdir(parents=True, exist_ok=True)
    (out / "calibration.json").write_text(json.dumps(base, indent=2))
    print(json.dumps({"images": base["vuln_images"], "scans": base["study14_scans"]}, indent=2))


if __name__ == "__main__":
    main()
