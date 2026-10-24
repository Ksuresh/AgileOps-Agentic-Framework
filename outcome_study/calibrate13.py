"""Study 13 calibration: select low-risk images by a fixed rule (no action outcomes measured).
Candidates: Online Boutique frontend tags from the stable image's registry, newest first (stable excluded). Keep tags with
0 critical CVEs and >= 1 HIGH/MEDIUM finding that roll out and serve HTTP 200. L1 = first, L2 = second; if fewer than two
qualify, the stable image fills the remaining slot (if it has >= 1 HIGH/MEDIUM finding)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from outcome_study import testbed as T  # noqa: E402
from outcome_study.calibrate import serves  # noqa: E402

TAGS = ["v0.10.5", "v0.10.4", "v0.10.3", "v0.10.2", "v0.10.1", "v0.10.0", "v0.9.0", "v0.8.1"]


def main() -> None:
    T.save_base()
    stable = T.base()["stable_image"]
    repo = stable.rsplit(":", 1)[0]
    scans, chosen = [], []
    for tag in TAGS:
        img = f"{repo}:{tag}"
        rec = {"image": img}
        try:
            rec["critical_cves"] = T.critical_cves(img)
            rec["noncritical_cves"] = T.noncritical_cves(img)
        except Exception as exc:
            rec["error"] = str(exc)[:200]
            scans.append(rec)
            continue
        if rec["critical_cves"] == 0 and rec["noncritical_cves"] >= 1 and len(chosen) < 2:
            rec["rollout_ok"] = T.apply(T.template(["vuln"], {"vuln": "X"}, {"X": img}), 1)
            rec["serves_http_200"] = serves()
            if rec["rollout_ok"] and rec["serves_http_200"]:
                chosen.append(img)
        scans.append(rec)
    st = {"image": stable, "critical_cves": T.critical_cves(stable), "noncritical_cves": T.noncritical_cves(stable)}
    while len(chosen) < 2 and st["critical_cves"] == 0 and st["noncritical_cves"] >= 1 and stable not in chosen:
        chosen.append(stable)
    T.apply(T.template([], {}, {}), 1)
    base = json.loads((ROOT / "outcome_study12" / "calibration" / "calibration.json").read_text())
    base["vuln_images"].update({"L1": chosen[0] if chosen else None, "L2": chosen[1] if len(chosen) > 1 else None})
    base["study13_scans"] = scans + [dict(st, note="stable")]
    base["note"] = "Study 13: Study 9 thresholds; images A-D as Studies 9/12; L1, L2 low-risk images selected by calibrate13.py."
    out = ROOT / "outcome_study13" / "calibration"
    out.mkdir(parents=True, exist_ok=True)
    (out / "calibration.json").write_text(json.dumps(base, indent=2))
    print(json.dumps({"L": [base["vuln_images"]["L1"], base["vuln_images"]["L2"]], "stable": st, "scans": scans}, indent=2))


if __name__ == "__main__":
    main()
