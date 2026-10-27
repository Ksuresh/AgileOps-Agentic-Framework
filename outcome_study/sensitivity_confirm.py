"""Oracle-threshold sensitivity for the confirmatory studies 14-16 (same 27-setting grid as sensitivity.py): for each
setting, O1 accuracy on fault cases of AAF's v1.4 rules (B1-14; identical to AAF-14's action on every full-evidence
case observed) and of every comparison decider, and whether the rules stay at least as accurate as / more accurate than
each. Reads committed measurements and decisions only (including decisions_gpt). Output: outcome_study/sensitivity_confirm.json."""
from __future__ import annotations

import itertools
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from outcome_study.sensitivity import STATE_OF, decisions, o1_sets  # noqa: E402

COMP = ["D-DevOps-14", "D-SRE-14", "D-FinOps-14", "D-Sec-14", "D-DevOps", "D-SRE", "D-FinOps", "D-Sec",
        "B5", "B6", "B5-GPT", "B6-GPT", "AlwaysRollback", "Dominant", "BlockAnyCVE"]


def main() -> None:
    grid = list(itertools.product((1.5, 2.0, 3.0), (0.5, 1.0, 2.0), (2.0, 3.0, 4.0)))
    res = {}
    for s in ("outcome_study14", "outcome_study15", "outcome_study16"):
        sd = ROOT / s
        dec = decisions(sd)
        g = sd / "decisions_gpt" / "gpt_rows.jsonl"
        if g.exists():
            tmp = defaultdict(lambda: defaultdict(list))
            for line in g.read_text().splitlines():
                r = json.loads(line)
                if r.get("error") is None:
                    tmp[r["decider"]][r["case"]].append(r["action"])
            dec.update({d: {c: Counter(v).most_common(1)[0][0] for c, v in cs.items()} for d, cs in tmp.items()})
        comps = [c for c in COMP if c in dec]
        rows = []
        for rf, et, cf in grid:
            o1 = o1_sets(sd, rf, et, cf)
            faults = [c for c in o1 if c.split("-")[1][0] in "RPSKDX"]
            acc = {d: sum(STATE_OF.get(dec[d].get(c)) in o1[c] for c in faults if c in dec[d]) for d in ["B1-14"] + comps}
            n = {d: sum(1 for c in faults if c in dec[d]) for d in ["B1-14"] + comps}
            rows.append({"rel_factor": rf, "err_pct": et, "cost_factor": cf, "n_faults": len(faults), "correct": acc, "n": n})
        res[s] = {"settings": len(rows), "rules_min": min(r["correct"]["B1-14"] for r in rows), "rules_max": max(r["correct"]["B1-14"] for r in rows),
                  "n_faults": rows[0]["n_faults"],
                  "settings_rules_ge": {d: sum(r["correct"]["B1-14"] / r["n"]["B1-14"] >= r["correct"][d] / max(1, r["n"][d]) for r in rows) for d in comps},
                  "settings_rules_gt": {d: sum(r["correct"]["B1-14"] / r["n"]["B1-14"] > r["correct"][d] / max(1, r["n"][d]) for r in rows) for d in comps},
                  "rows": rows}
        print(s, res[s]["rules_min"], res[s]["rules_max"], "/", res[s]["n_faults"], {d: res[s]["settings_rules_gt"][d] for d in comps})
    (ROOT / "outcome_study" / "sensitivity_confirm.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
