"""AAF v3 text-side runner (PLAN.md). Same evidence handling as postmortems/run_study.py.

Per case and run: one numeric-reader call (frozen v2 prompt) and one categorical-reader call (v3 prompt).
  L   = frozen AAF v1 on the numeric extraction (identical to the postmortem study's arm L)
  V3  = AAF v3 on the same numeric extraction plus the categorical facts (paired with L by construction)
  A   = AI-only (frozen prompt from run_study.py), optional
Usage:
  python -m external_validation.aaf_v3.run_text_v3 --split dev            # 40 development postmortems
  python -m external_validation.aaf_v3.run_text_v3 --split holdout --arms L V3 A
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from external_validation.aaf_v3.text_v3 import CAT_PROMPT, CAT_SPECS, decide_v3
from external_validation.postmortems import run_study as S
from text_evidence_pilot.llm_reader import READER_SYSTEM_V2
from text_evidence_pilot.run_pilot import fill
from text_evidence_pilot.schema import FIELD_SPECS

HERE = Path(__file__).resolve().parent
PM = ROOT / "external_validation" / "postmortems"
SPLITS = {
    "dev": {"labels": [PM / "labels_rater1.csv", PM / "labels_rater1_gitlab.csv"], "hashes": PM / "evidence_hashes.json"},
    "holdout": {"labels": [HERE / "holdout" / "labels_rater1_holdout.csv"], "hashes": HERE / "holdout" / "evidence_hashes.json"},
}


def load(split: str) -> list[dict]:
    S.LABEL_FILES = SPLITS[split]["labels"]
    S.HASHES = SPLITS[split]["hashes"]
    return S.load_cases()


def run_one(llm: S.LLM, c: dict, arm: str, run: int) -> dict:
    case = S.pm_case(c)
    docs = {d["id"]: d["text"] for d in case["documents"]}
    if arm == "A":
        return {**S.run_one(llm, c, "A", run), "arm": "A"}
    eff = os.environ.get("OPENAI_REASONING_EFFORT", "none")
    num = llm.call(READER_SYSTEM_V2, S.reader_payload(case, False), eff)
    exs = list(num["obj"].get("extractions") or []) if not llm.mock else S.mock_extractions(case, False)
    ev, info = fill(case, [x for x in exs if x.get("field") in FIELD_SPECS], S.VERSION)
    a_l = S.decide(ev)
    cat = llm.call(CAT_PROMPT, {"fact_catalogue": CAT_SPECS, "documents": [{"id": d["id"], "text": d["text"]} for d in case["documents"]]},
                   os.environ.get("CAT_REASONING_EFFORT", "low"))
    facts = list(cat["obj"].get("facts") or []) if not llm.mock else []
    a_v3, info3 = decide_v3(ev, info["accepted"], facts, docs)
    return {"case": c["id"], "arm": "LV3", "run": run, "action_L": a_l, "action_V3": a_v3, "numeric": info, "v3": info3,
            "raw_numeric": num["raw"], "raw_cat": cat["raw"], "error": num["error"] or cat["error"],
            "usage": {"total_tokens": num["usage"].get("total_tokens", 0) + cat["usage"].get("total_tokens", 0)}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=list(SPLITS), required=True)
    ap.add_argument("--arms", nargs="*", default=["L", "V3"])
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--source-json", nargs="*")
    args = ap.parse_args()

    cases = load(args.split)
    S.attach_evidence(cases, args.source_json)
    usable = [c for c in cases if c["evidence_match"]]
    llm = S.LLM(args.mock)
    jobs = [(c, "LV3", r) for c in usable for r in range(args.repeats)]
    if "A" in args.arms:
        jobs += [(c, "A", r) for c in usable for r in range(args.repeats)]
    with ThreadPoolExecutor(max_workers=1 if args.mock else 8) as pool:
        runs = list(pool.map(lambda j: run_one(llm, *j), jobs))

    per = defaultdict(lambda: defaultdict(list))
    for x in runs:
        if x["arm"] == "A":
            per[x["case"]]["A"].append(x["action"])
        else:
            per[x["case"]]["L"].append(x["action_L"]); per[x["case"]]["V3"].append(x["action_V3"])
    labels = Counter(c["label"] for c in usable)
    majority = labels.most_common(1)[0][0]
    arms = ["M0"] + [a for a in ("L", "V3", "A") if a in args.arms]
    rows = []
    for c in cases:
        row = {"case": c["id"], "source": c["source"], "label": c["label"], "evidence_match": c["evidence_match"]}
        if c["evidence_match"]:
            adm = S.ADMISSIBLE[c["label"]]
            row.update({"M0_action": majority, "M0_correct": majority in adm})
            for a in arms[1:]:
                acts = per[c["id"]][a]
                k = sum(x in adm for x in acts)
                row.update({f"{a}_correct": k * 2 > len(acts), f"{a}_runs_correct": f"{k}/{len(acts)}",
                            f"{a}_modal_action": Counter(acts).most_common(1)[0][0], f"{a}_actions": json.dumps(dict(Counter(acts)), sort_keys=True)})
            lv = [x for x in runs if x["case"] == c["id"] and x["arm"] == "LV3"]
            row["V3_facts"] = json.dumps(dict(Counter(f for x in lv for f in x["v3"]["facts_accepted"])), sort_keys=True)
            row["V3_rule_fired_runs"] = sum(x["v3"]["rollback_rule_fired"] for x in lv)
        rows.append(row)
    ok = [r for r in rows if r["evidence_match"]]
    col = lambda a, rs=ok: [bool(r[f"{a}_correct"]) for r in rs]
    tests = {f"V3_vs_{b}": S.mcnemar(col("V3"), col(b)) for b in arms if b != "V3"}
    primary = {k: tests[k] for k in ("V3_vs_L", "V3_vs_M0")}
    summary = {
        "plan": "external_validation/aaf_v3/PLAN.md", "split": args.split, "mock": args.mock, "repeats": args.repeats,
        "cases": len(ok), "excluded_hash_mismatch": [r["case"] for r in rows if not r["evidence_match"]],
        "label_distribution": dict(labels), "majority_label": majority,
        "agreement": {a: S.rate(col(a)) for a in arms},
        "paired": tests, "primary_holm": S.holm({k: v["exact_p"] for k, v in primary.items()}),
        "by_label": {lab: {a: S.rate(col(a, [r for r in ok if r["label"] == lab])) for a in arms} for lab in sorted(labels)},
        "modal_actions": {a: dict(Counter(r[f"{a}_modal_action"] for r in ok)) for a in arms[1:]},
        "non_decision_modal": {a: S.rate([r[f"{a}_modal_action"] in S.NON_DECISIONS for r in ok]) for a in arms[1:]},
        "wrong_committed_modal": {a: sum(1 for r in ok if r[f"{a}_modal_action"] not in S.NON_DECISIONS and not r[f"{a}_correct"]) for a in arms[1:]},
        "v3_facts_rejected": dict(Counter(y["reason"] for x in runs if x["arm"] == "LV3" for y in x["v3"]["facts_rejected"])),
        "failed_calls": sum(1 for x in runs if x["error"]),
    }
    v3_files = [HERE / "text_v3.py", HERE / "run_text_v3.py"]
    meta = {"created_utc": datetime.now(timezone.utc).isoformat(), "model": None if args.mock else llm.model,
            "reader_effort": os.environ.get("OPENAI_REASONING_EFFORT", "none"), "cat_effort": os.environ.get("CAT_REASONING_EFFORT", "low"),
            "v3_sha256": {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in v3_files},
            "cat_prompt_sha256": hashlib.sha256(CAT_PROMPT.encode()).hexdigest(),
            "github_run_id": os.environ.get("GITHUB_RUN_ID"), "git_sha": os.environ.get("GITHUB_SHA")}
    out = HERE / "results" / args.split / ("mock" if args.mock else datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    (out / "metadata.json").write_text(json.dumps(meta, indent=2))
    with (out / "case_results.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(dict.fromkeys(k for r in rows for k in r))); w.writeheader(); w.writerows(rows)
    raw = out / "raw_runs.jsonl"
    raw.write_text("\n".join(json.dumps(x, default=str) for x in runs))
    S.encrypt(raw, out / "raw_runs.jsonl")
    print(json.dumps({k: summary[k] for k in ("cases", "agreement", "paired", "non_decision_modal", "wrong_committed_modal")}, indent=1))


if __name__ == "__main__":
    main()
