"""Study 13 descriptive analysis (PROTOCOL.md): E13.1 low-risk releases continue (O2-correct); E13.2 critical CVEs removed (O1)."""
from __future__ import annotations

import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault("OUTCOME_DIR", "outcome_study13")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from outcome_study import analyze as A  # noqa: E402

HERE = ROOT / "outcome_study13"


def main() -> None:
    table = A.outcome_table(json.loads((HERE / "calibration" / "calibration.json").read_text()))
    dec = defaultdict(lambda: defaultdict(list))
    for line in (HERE / "decisions" / "deterministic.jsonl").read_text().splitlines():
        r = json.loads(line); dec[r["decider"]][r["case"]].append(r["action"])
    for f in sorted((HERE / "decisions").glob("llm_shard_*.jsonl")):
        for line in f.read_text().splitlines():
            r = json.loads(line)
            if r.get("error") is None:
                dec[r["decider"]][r["case"]].append(r["action"])

    def maj(xs):
        c = Counter(xs).most_common()
        return sorted(a for a, n in c if n == c[0][1])[0] if c else None
    lr = [c for c in table if c.startswith("LR")]
    cr = [c for c in table if c.startswith("CR")]
    res = {"oracle": {c: {k: table[c][k] for k in ("O1", "O2")} for c in table}, "E13_1_lowrisk_continue_O2": {}, "E13_2_critical_removed_O1": {},
           "E13_2_critical_O2": {}, "actions": {}}
    for d in sorted(dec):
        res["E13_1_lowrisk_continue_O2"][d] = f"{sum(A.STATE_OF.get(maj(dec[d].get(c) or [])) in table[c]['O2'] for c in lr)}/{len(lr)}"
        res["E13_2_critical_removed_O1"][d] = f"{sum(A.STATE_OF.get(maj(dec[d].get(c) or [])) in table[c]['O1'] for c in cr)}/{len(cr)}"
        res["E13_2_critical_O2"][d] = f"{sum(A.STATE_OF.get(maj(dec[d].get(c) or [])) in table[c]['O2'] for c in cr)}/{len(cr)}"
        res["actions"][d] = {c: maj(dec[d].get(c) or []) for c in sorted(dec[d])}
    (HERE / "analysis.json").write_text(json.dumps(res, indent=2))
    for k in ("E13_1_lowrisk_continue_O2", "E13_2_critical_removed_O1", "E13_2_critical_O2"):
        print(k, res[k])


if __name__ == "__main__":
    main()
