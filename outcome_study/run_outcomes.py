"""Study 9 outcome runner: for each case, build the incident, record the snapshot (decider input), then for every
state-changing action rebuild the incident, apply the action and measure the outcome. Sharded by SHARD / NSHARDS."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from outcome_study import testbed as T  # noqa: E402

HERE = ROOT / os.environ.get("OUTCOME_DIR", "outcome_study")


def main() -> None:
    cfg = yaml.safe_load((HERE / "cases.yaml").read_text())
    cal = json.loads((HERE / "calibration" / "calibration.json").read_text())
    images = cal["vuln_images"]
    assert images["A"] and images["B"], "calibration did not select two vulnerable images"
    shard, nshards = int(os.environ.get("SHARD", "0")), int(os.environ.get("NSHARDS", "1"))
    rep = os.environ.get("REP", "1")
    only = [x for x in os.environ.get("CASES", "").split() if x]
    cases = [c for c in cfg["cases"] if not only or c["id"] in only][shard::nshards]
    T.save_base()
    for case in cases:
        load = int(case["high"]) if case["load"] == "high" else int(cfg["loads"][case["load"]])
        d = HERE / "measurements" / f"rep{rep}"
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"{case['id']}.json"
        if path.exists():
            continue
        t0 = time.time()
        T.apply(T.template(case["previous"], case, images), case["replicas"])
        ok = T.apply(T.template(case["current"], case, images), case["replicas"])
        snap = T.measure(load)
        snap["rollout_ok"] = ok
        snap.update(T.release_info())
        outcomes = {}
        for action in T.ACTION_STATES:
            T.apply(T.template(case["previous"], case, images), case["replicas"])
            T.apply(T.template(case["current"], case, images), case["replicas"])
            mods, replicas = T.action_release(action, case, snap)
            ok = T.apply(T.template(mods, case, images), replicas)
            m = T.measure(load)
            m["rollout_ok"] = ok
            outcomes[action] = m
            print(case["id"], action, m["http"]["latency_p95_ms"], m["http"]["error_rate_pct"], m["footprint_cpu_m"],
                  m["critical_cves_new"], m["secret_in_config"], flush=True)
        path.write_text(json.dumps({"case": case, "load": load, "rep": rep, "snapshot": snap, "outcomes": outcomes,
                                    "wall_s": round(time.time() - t0, 1)}, indent=1))
    T.apply(T.template([], {}, images), 1)


if __name__ == "__main__":
    main()
