"""Post hoc (after Study 14 results): recompute the rule-based comparison deciders of Study 14 with rule set v1.4 instead of
v1.0, from the committed snapshots and B5 tool traces (no new model calls). Pre-specified deciders used v1.0 domain rules for
the siloed experts, B2 and B3; this script reports how the comparisons change if they use v1.4."""
from __future__ import annotations

import copy
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["OUTCOME_DIR"] = "outcome_study14"
from outcome_study import analyze as A  # noqa: E402
from outcome_study import decide as D  # noqa: E402
from agentic_experiments import agentic_runtime as AR  # noqa: E402
from agentic_experiments.hybrid import missing_fields  # noqa: E402
from agentic_experiments.rules_v14 import deterministic_assessment_v14 as V14  # noqa: E402
from external_validation.expert_study.analyze import holm, mcnemar  # noqa: E402

HERE = ROOT / "outcome_study14"


def main() -> None:
    table = A.outcome_table(json.loads((HERE / "calibration" / "calibration.json").read_text()))
    variants = json.loads((HERE / "decisions" / "variants.json").read_text())
    dec = defaultdict(dict)
    for v in variants:
        ev = v["evidence"]
        for name, block in D.BLOCK_OF.items():
            dec[name + "(v1.4)"][v["id"]] = V14({block: copy.deepcopy(ev[block])})["selected_action"]
        e2 = copy.deepcopy(ev)
        for f in missing_fields(e2):
            if D.F2T.get(f) == v.get("expected_tool"):
                e2 = AR.merge_tool_result(e2, copy.deepcopy(v["tool_result"]))
                break
        dec["B2(v1.4)"][v["id"]] = V14(e2)["selected_action"]
    b3 = defaultdict(list)
    for f in sorted((HERE / "decisions").glob("llm_shard_*.jsonl")):
        for line in f.read_text().splitlines():
            r = json.loads(line)
            if r.get("error") or r["decider"] != "B5":
                continue
            v = next(x for x in variants if x["id"] == r["case"])
            ev = copy.deepcopy(v["evidence"])
            for t in r.get("tool_events") or []:
                if t.get("tool") == v.get("expected_tool") and isinstance(t.get("result"), dict):
                    ev = AR.merge_tool_result(ev, t["result"])
            b3[r["case"]].append(V14(ev)["selected_action"])
    dec["B3(v1.4)"] = {c: Counter(a).most_common(1)[0][0] for c, a in b3.items()}
    main_an = json.loads((HERE / "analysis.json").read_text())["per_case"]
    dec["AAF-14"] = {c: x["AAF-14"] for c, x in main_an.items()}
    ok = lambda d, c: A.STATE_OF.get(dec[d].get(c)) in table[c.replace("-W", "")]["O1"]
    grp = lambda c: c.split("-")[1][0]
    faults = [c for c in table if grp(c) in "RPSKDX"]
    withheld = [c + "-W" for c in table if (c + "-W") in dec["B2(v1.4)"]]
    res = {"faults_n": len(faults), "withheld_n": len(withheld),
           "faults_correct": {d: sum(ok(d, c) for c in faults) for d in dec},
           "withheld_correct": {d: sum(ok(d, c) for c in withheld) for d in dec if all(c in dec[d] for c in withheld)}}
    silos = [d for d in dec if d.startswith("D-")]
    t = {d: mcnemar({c: ok("AAF-14", c) for c in faults}, {c: ok(d, c) for c in faults}, faults) for d in silos}
    for (d, v), p in zip(t.items(), holm([v["p_exact"] for v in t.values()])):
        v["p_holm"] = p
    res["AAF14_vs_v14_siloed_rules"] = t
    res["AAF14_vs_B2v14_withheld"] = mcnemar({c: ok("AAF-14", c) for c in withheld}, {c: ok("B2(v1.4)", c) for c in withheld}, withheld)
    res["AAF14_vs_B3v14_withheld"] = mcnemar({c: ok("AAF-14", c) for c in withheld}, {c: ok("B3(v1.4)", c) for c in withheld}, withheld)
    (HERE / "posthoc_v14_baselines.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
