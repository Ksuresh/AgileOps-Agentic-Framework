"""Discriminating-case analysis for every outcome study (no model calls). A fault case discriminates when not every one
of the five actions is O1-correct; on the others every decider scores a point. Reports, per study, how many fault cases
discriminate and each decider's O1 accuracy on all fault cases and on the discriminating ones, with exact McNemar tests of
the study's primary AAF arm against the siloed experts and the LLM deciders on the discriminating cases."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from outcome_study.analyze_gpt import load, tests  # noqa: E402

STUDIES = {  # study -> (fault-case predicate, primary AAF arm)
    "outcome_study": (lambda c: c.startswith(("OS", "OX")), "AAF"),
    "outcome_study12": (lambda c: not c.startswith("PC"), "AAF-12"),
    "outcome_study13": (lambda c: True, "AAF-12"),
    "outcome_study14": (lambda c: c.split("-")[1][0] in "RPSKDX", "AAF-14"),
    "outcome_study15": (lambda c: c.split("-")[1][0] in "RPSKDX", "AAF-14"),
    "outcome_study16": (lambda c: c.split("-")[1][0] in "RPSKDX", "AAF-14"),
}
SHOW = ["AAF", "AAF-12", "AAF-14", "B1", "B1-14", "D-DevOps", "D-SRE", "D-FinOps", "D-Sec", "D-DevOps-14", "D-SRE-14", "D-FinOps-14",
        "D-Sec-14", "L-DevOps", "L-SRE", "L-FinOps", "L-Sec", "B5", "B6", "B5-GPT", "B6-GPT", "AlwaysRollback", "AlwaysNone", "BlockAnyCVE"]


def main() -> None:
    res = {}
    for st, (is_fault, primary) in STUDIES.items():
        if not (ROOT / st / "decisions" / "deterministic.jsonl").exists():
            continue
        A, table, dec, ok, maj, _ = load(st)
        if primary not in dec:
            primary = "AAF"
        faults = [c for c in sorted(table) if is_fault(c)]
        disc = [c for c in faults if len(table[c]["O1"]) < 5]
        names = [d for d in SHOW if d in dec]
        siloed = [d for d in ("D-DevOps", "D-SRE", "D-FinOps", "D-Sec", "D-DevOps-14", "D-SRE-14", "D-FinOps-14", "D-Sec-14") if d in dec]
        llm = [d for d in ("B5", "B6", "B5-GPT", "B6-GPT", "AlwaysRollback") if d in dec]
        res[st] = {"primary": primary, "n_faults": len(faults), "n_discriminating": len(disc),
                   "non_discriminating": [c for c in faults if c not in disc],
                   "correct_all_faults": {d: sum(ok(d, c) for c in faults) for d in names},
                   "correct_discriminating": {d: sum(ok(d, c) for c in disc) for d in names},
                   "tests_discriminating": {"siloed": tests(ok, primary, siloed, disc), "llm_and_policies": tests(ok, primary, llm, disc)}}
    (ROOT / "outcome_study" / "discriminating.json").write_text(json.dumps(res, indent=2))
    for st, r in res.items():
        print(st, r["primary"], r["n_faults"], r["n_discriminating"], {d: r["correct_discriminating"][d] for d in r["correct_discriminating"]
                                                                       if d in (r["primary"], "D-Sec", "B5", "B6", "AlwaysRollback")})


if __name__ == "__main__":
    main()
