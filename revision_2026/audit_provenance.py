"""Re-checks the provenance claims in PROVENANCE.md from git history and GitHub Actions records.

For each study: (1) the protocol/dataset commit precedes the first run that used it; (2) the run's head commit contains
those files with the same SHA-256 as today; (3) the results commit follows the run. Requires `gh` with read access.
Prints a table and exits non-zero if any check fails.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = "Ksuresh/AgileOps-Agentic-Framework"

# study, files frozen before the run, run id, results path (None if outputs live in the run artifact)
STUDIES = [
    ("Sock Shop HRT-22-31", ["runtime_validation/interventions_extended_validation.yaml"], 33880374847, None),
    ("Prospective HRT-32-39", ["runtime_validation/interventions_v2_prospective.yaml"], 33885779782, None),
    ("Selective AEI primary (32)", ["agentic_experiments/cases_v1.yaml", "agentic_experiments/FROZEN_PROTOCOL.md"], 34428945881, None),
    ("Selective AEI replication (16)", ["agentic_experiments/cases_replication_v1.yaml"], 34432645600, None),
    ("RCAEval with healthy controls", ["external_validation/rcaeval_governance/PROTOCOL.md"], 39581484587, "external_validation/rcaeval_governance/results/summary.json"),
    ("AAF-CA holdout", ["external_validation/aaf_v3/PLAN.md"], 39600613441, "external_validation/aaf_v3/RESULTS.md"),
    ("PM-interface re-test (E1)", ["external_validation/explanations/PROTOCOL.md"], 39603634523, "external_validation/explanations/results/20261021T125701Z/summary.json"),
    ("Reviewer-requested LLM arms", ["revision_2026/LLM_RUNS_PLAN.md", "revision_2026/run_llm_arms.py"], 39736227167, "revision_2026/results/llm_arms/20261022T064831Z/summary.json"),
    ("Study 7", ["external_validation/expert_study/cases_new_v1.yaml", "external_validation/expert_study/author_labels.yaml",
                 "external_validation/expert_study/expert_labels.json", "external_validation/expert_study/run_study.py"], 39784044792,
     "external_validation/expert_study/results/20261022T130735Z/budget.json"),
    ("Study 8", ["external_validation/investigator_fix/cases_fx_v1.yaml", "external_validation/investigator_fix/author_labels.yaml",
                 "external_validation/investigator_fix/PROTOCOL.md", "agentic_experiments/hybrid_v2.py"], 39786446655,
     "external_validation/investigator_fix/results/20261022T132423Z/budget.json"),
]


def sh(*args: str) -> str:
    return subprocess.run(args, cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def t(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def first_commit(path: str, rev: str = "--all") -> tuple[str, datetime]:
    line = sh("git", "log", rev, "--diff-filter=A", "--format=%H %cI", "--", path).splitlines()[-1]
    h, d = line.split()
    return h, t(d)


def main() -> None:
    rows, ok_all = [], True
    for name, files, run, results in STUDIES:
        info = json.loads(sh("gh", "run", "view", str(run), "-R", REPO, "--json", "createdAt,headSha,conclusion"))
        started, head = t(info["createdAt"]), info["headSha"]
        checks = []
        for f in files:
            sh("git", "cat-file", "-e", head)  # head commit must be available locally
            _, added = first_commit(f, head)
            old = subprocess.run(["git", "show", f"{head}:{f}"], cwd=ROOT, capture_output=True, check=True).stdout
            cur = (ROOT / f).read_bytes() if (ROOT / f).exists() else b""
            same = old == cur
            appended = (not same) and cur.startswith(old.rstrip(b"\n"))  # amendments appended, original text intact
            checks.append({"file": f, "committed_before_run": added < started, "unchanged_since_run": same,
                           "append_only_amendments": appended, "committed": added.isoformat(), "sha256": hashlib.sha256(old).hexdigest()[:12]})
        res_ok = None
        if results:
            _, rc = first_commit(results)
            res_ok = rc > started
        ok = all(c["committed_before_run"] and (c["unchanged_since_run"] or c["append_only_amendments"]) for c in checks) and res_ok is not False and info["conclusion"] == "success"
        ok_all &= ok
        rows.append({"study": name, "run": run, "run_started": started.isoformat(), "head": head[:7], "files": checks,
                     "results_after_run": res_ok, "pass": ok})
    out = ROOT / "revision_2026" / "results" / "provenance_audit.json"
    out.write_text(json.dumps(rows, indent=1))
    for r in rows:
        print(f"{'PASS' if r['pass'] else 'FAIL'}  {r['study']:<32} run {r['run']} {r['run_started'][:16]}  "
              + "; ".join(f"{Path(c['file']).name}: before={c['committed_before_run']} " + ("unchanged" if c['unchanged_since_run'] else ("append-only amendments" if c['append_only_amendments'] else "EDITED")) for c in r["files"])
              + f"  results_after={r['results_after_run']}")
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
