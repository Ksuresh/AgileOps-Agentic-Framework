"""Amendment A8: independent AI second rater for the answer key (no human rater available).

For each of the 40 included cases the rater sees the responders' timeline window entries[start:cut+4],
numbered from 1, with no rater-1 information, and returns (a) the number of the first deliberate human
response action or "exclude", (b) the rubric label. Three independent calls per case; the majority is the
rater-2 answer. Agreement with rater 1: label agreement and Cohen's kappa (exclude counted as its own
category), and first-action entry agreement. Text is fetched at run time and checked against the frozen
evidence hashes (as in run_study.py); raw outputs are committed encrypted only.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from external_validation.postmortems.run_study import HASHES, LLM, encrypt, fetch_text, load_cases
from external_validation.postmortems.timeline_extract import ordered_entries

HERE = Path(__file__).resolve().parent
LABELS = ["Rollback", "Patch", "Scale", "Mitigate", "exclude"]
SHORT = {"Rollback to stable deployment": "Rollback", "Patch or block release": "Patch",
         "Scale adjustment": "Scale", "Mitigate and monitor": "Mitigate"}
EXTRA = 4
RATER_SYSTEM = """You are an independent rater building an answer key from real incident timelines.
You receive numbered timeline entries written by the incident responders. Identify the FIRST deliberate human
response action (an action taken to change the system, not investigation, paging, communication or status updates),
and classify it with this rubric:
- Rollback: revert, roll back, or disable/undo a change deployed shortly before the impact
- Patch: halt or pause deploys/releases, release freeze, ship a hotfix or patch
- Scale: add capacity, scale out or up
- Mitigate: restart, failover, reroute or shed traffic, rate-limit, disable a feature not just deployed, run a repair procedure
Answer "exclude" if no entry states such an action explicitly, if no observation of the system precedes it, if the first
effective action was automatic, or if the incident is a security breach, human-error data loss, hardware/power/facility
failure, or purely an upstream provider outage.
Return only strict JSON: {"first_action_entry": <entry number or null>, "label": "Rollback"|"Patch"|"Scale"|"Mitigate"|"exclude", "reason": <one sentence>}
"""


def kappa(a: list[str], b: list[str]) -> float | None:
    n = len(a)
    if not n:
        return None
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb[k] for k in set(a) | set(b)) / (n * n)
    return None if pe == 1 else round((po - pe) / (1 - pe), 3)


def main() -> None:
    frozen = json.loads(HASHES.read_text())
    cases = load_cases()
    llm = LLM(mock=False)
    effort = os.environ.get("RATER_REASONING_EFFORT", "high")
    work = []
    for c in cases:
        _, text = fetch_text(c)
        items = ordered_entries(text)
        ev = items[c["start"]:c["cut"]]
        c["evidence_match"] = hashlib.sha256("\n".join(ev).encode()).hexdigest() == frozen[c["id"]]["evidence_sha256"]
        c["window"] = items[c["start"]:c["cut"] + EXTRA]
        if c["evidence_match"]:
            work += [(c, k) for k in range(3)]

    def one(job):
        c, k = job
        payload = {"timeline_entries": [{"n": i + 1, "text": e[:1200]} for i, e in enumerate(c["window"])]}
        res = llm.call(RATER_SYSTEM, payload, effort)
        lab = str(res["obj"].get("label", ""))
        ent = res["obj"].get("first_action_entry")
        return {"case": c["id"], "k": k, "label": lab if lab in LABELS else "INVALID",
                "entry": int(ent) if isinstance(ent, (int, float)) else None, "raw": res["raw"], "error": res["error"]}

    with ThreadPoolExecutor(max_workers=8) as pool:
        runs = list(pool.map(one, work))

    rows = []
    for c in cases:
        rs = [x for x in runs if x["case"] == c["id"]]
        if not rs:
            rows.append({"case": c["id"], "evidence_match": False}); continue
        lab = Counter(x["label"] for x in rs).most_common(1)[0][0]
        ents = Counter(x["entry"] for x in rs if x["label"] == lab).most_common(1)
        ent = ents[0][0] if ents else None
        r1_entry = c["cut"] - c["start"] + 1  # rater-1 first action, numbered from 1 in the window
        rows.append({"case": c["id"], "source": c["source"], "evidence_match": True, "rater1_label": SHORT[c["label"]],
                     "rater2_label": lab, "rater2_votes": json.dumps(dict(Counter(x["label"] for x in rs))),
                     "label_agree": lab == SHORT[c["label"]], "rater1_entry": r1_entry, "rater2_entry": ent,
                     "entry_agree": ent == r1_entry})
    ok = [r for r in rows if r["evidence_match"]]
    a, b = [r["rater1_label"] for r in ok], [r["rater2_label"] for r in ok]
    both = [r for r in ok if r["rater2_label"] != "exclude"]
    summary = {
        "amendment": "A8", "model": llm.model, "reasoning_effort": effort, "calls": len(runs),
        "failed_calls": sum(1 for x in runs if x["error"]), "cases": len(ok),
        "rater2_excluded": [r["case"] for r in ok if r["rater2_label"] == "exclude"],
        "label_agreement_all": f"{sum(r['label_agree'] for r in ok)}/{len(ok)}",
        "cohens_kappa_all_with_exclude": kappa(a, b),
        "label_agreement_both_include": f"{sum(r['label_agree'] for r in both)}/{len(both)}",
        "cohens_kappa_both_include": kappa([r["rater1_label"] for r in both], [r["rater2_label"] for r in both]),
        "first_action_entry_agreement_both_include": f"{sum(r['entry_agree'] for r in both)}/{len(both)}",
        "confusion_rater1_to_rater2": dict(Counter(f"{r['rater1_label']}->{r['rater2_label']}" for r in ok)),
        "rater_prompt_sha256": hashlib.sha256(RATER_SYSTEM.encode()).hexdigest(),
        "github_run_id": os.environ.get("GITHUB_RUN_ID"),
    }
    out = HERE / "rater2" / "ai_rater" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    with (out / "case_agreement.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(dict.fromkeys(k for r in rows for k in r))); w.writeheader(); w.writerows(rows)
    raw = out / "raw.jsonl"
    raw.write_text("\n".join(json.dumps(x) for x in runs))
    encrypt(raw, out / "raw.jsonl")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
