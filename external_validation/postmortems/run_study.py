"""Real-postmortem governance study — see PROTOCOL.md (pre-registered, amendments A1-A6).

Answer key: the first action the original responders took (labels_rater1*.csv, frozen before
any arm ran). Evidence: timeline entries[start:cut] only, re-fetched at run time and checked
against evidence_hashes.json (frozen). Plain postmortem text and verbatim quotes are never
committed: raw run logs are written encrypted with the screening public key.

Arms
  M0  majority class ("Mitigate and monitor" in the final set)
  R   regex reader -> checker v2 -> frozen AAF
  L   LLM reader v2 (frozen prompt) -> checker v2 -> frozen AAF                 (primary)
  A   AI-only: one LLM call, timeline + action list + AAF policy, reasoning medium
  L+  LLM reader v2 + categorical fields -> checker v2 -> AAF-ext (one added rule) (secondary)

Usage (repo root):
  python -m external_validation.postmortems.run_study --mock --source-json a.json b.json   # offline dry run
  OPENAI_API_KEY=... python -m external_validation.postmortems.run_study                   # real run (fetches)
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import platform
import re
import subprocess
import sys
import tempfile
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scipy.stats import binomtest

from agentic_experiments.agentic_runtime import ACTIONS
from agentic_experiments.hybrid import deterministic_assessment
from external_validation.postmortems.timeline_extract import ordered_entries, timeline_hash
from orchestrator.cross_domain_v2 import _material_sre_outcome, interaction_state_v2, TIER_HARD_OVERRIDE
from text_evidence_pilot import regex_reader
from text_evidence_pilot.llm_reader import READER_SYSTEM_V2, _json_obj
from text_evidence_pilot.run_pilot import fill
from text_evidence_pilot.schema import FIELD_SPECS, split
from text_evidence_pilot.validator import _context_reason, _norm

HERE = Path(__file__).resolve().parent
LABEL_FILES = [HERE / "labels_rater1.csv", HERE / "labels_rater1_gitlab.csv"]
HASHES = HERE / "evidence_hashes.json"
PUBKEY = HERE / "screening" / "screen_public.pem"
VERSION = "v2"
SCOPE = {"service": "the service or system affected by the incident described in the timeline"}
DOC_ID = "TIMELINE"

ROLLBACK, PATCH, SCALE, MITIGATE = ("Rollback to stable deployment", "Patch or block release",
                                    "Scale adjustment", "Mitigate and monitor")
# Amendment A6: AAF has two release-halting labels; both express the rubric's "halt deploys / hotfix" row.
ADMISSIBLE = {ROLLBACK: {ROLLBACK}, PATCH: {PATCH, "Block release and fix pipeline"},
              SCALE: {SCALE}, MITIGATE: {MITIGATE}}
MAJORITY = MITIGATE  # fixed by amendment A5 (19 of 40)
NON_DECISIONS = {"No action (observe)", "Escalate for evidence/human review"}

# ---------------------------------------------------------------- L+ categorical fields (PROTOCOL "Secondary extension")
EXT_SPECS = {
    "sre.outage_present": {"type": "bool", "desc": "the service was down or unavailable"},
    "sre.error_spike": {"type": "bool", "desc": "errors or failures rose sharply"},
    "sre.latency_spike": {"type": "bool", "desc": "latency or slowness rose sharply"},
    "sre.resource_exhaustion": {"type": "bool", "desc": "CPU, memory, disk or connections ran out"},
    "deploy.change_preceded_impact": {"type": "bool", "desc": "a deployment, release or config change started shortly before the impact"},
}
EXT_MAP = {"sre.outage_present": ("sre.availability_pct", 94.0), "sre.error_spike": ("sre.error_rate_pct", 12.0),
           "sre.latency_spike": ("sre.p95_latency_ms", 800.0), "sre.resource_exhaustion": ("sre.saturation_pct", 90.0)}
EXT_PROMPT = READER_SYSTEM_V2 + """
Additional boolean fields are listed in categorical_fields. Set one to true only when the documents state it
for the scope service, with a verbatim quote and method "literal"; omit it otherwise (never output false
just because something is not mentioned).
"""

# ---------------------------------------------------------------- arm A prompt (B6-style AI-only)
A_POLICY = """Governance policy of the Agile-DevOps decision framework (choose exactly one action):
- Act only on evidence stated in the timeline. If no material problem is evidenced, choose "No action (observe)".
  If the evidence is too thin to justify any action, choose "Escalate for evidence/human review".
- Arbitration order: hard governance constraints first, then specific causal relations, then generic accumulation.
- Hard constraint: a failed release gate (pipeline failure, artifact mismatch, config drift) together with security
  evidence (critical vulnerabilities, policy violation, IAM drift) -> "Patch or block release".
- Specific causal relation: a recent deployment or change that is followed by material reliability degradation
  (errors, latency, unavailability, restart bursts) -> "Rollback to stable deployment".
- Specific causal relation: material reliability degradation together with resource/capacity pressure that more
  capacity would relieve -> "Scale adjustment".
- Release-integrity problems without user-facing degradation -> "Block release and fix pipeline".
- Cost or scaling inefficiency without reliability impact -> "Review scaling policy".
- Material reliability degradation without a deployment cause or capacity case -> "Mitigate and monitor"
  (restart, failover, reroute, rate-limit, disable a feature, repair).
"""
A_SYSTEM = """You are the decision-maker in a project-governance experiment for Agile-DevOps incident response.
You receive the incident timeline as recorded by responders up to (not including) their first response action.
Choose the governance action that should be taken next, using only the timeline and the governance policy.
Return only strict JSON: {"selected_action": <one of allowed_actions, verbatim>, "rationale_summary": <one or two sentences>}
"""


def prompt_hashes() -> dict[str, str]:
    h = lambda s: hashlib.sha256(s.encode()).hexdigest()
    return {"L_reader_v2": h(READER_SYSTEM_V2), "Lplus_reader": h(EXT_PROMPT), "A_system": h(A_SYSTEM), "A_policy": h(A_POLICY)}


# ---------------------------------------------------------------- data
def load_cases() -> list[dict[str, Any]]:
    out = []
    for f in LABEL_FILES:
        for r in csv.DictReader(f.open()):
            if r["decision"] == "include":
                out.append({"id": r["id"], "org": r["org"], "url": r["url"], "start": int(r["start"]), "cut": int(r["cut"]),
                            "label": r["action"], "timeline_sha256": r["timeline_sha256"],
                            "source": "gitlab" if r["id"].startswith("GL-") else "danluu"})
    return out


def fetch_text(case: dict[str, Any]) -> tuple[str, str]:
    if case["source"] == "gitlab":
        from external_validation.postmortems.screening import fetch_gitlab
        iid = int(case["url"].rstrip("/").split("/")[-1])
        n, text = fetch_gitlab.notes_text(iid)
        return ("ok" if n else "no_notes"), text
    from external_validation.postmortems.screening.fetch_encrypted import text
    return text(case["url"])


def attach_evidence(cases: list[dict[str, Any]], source_json: list[str] | None) -> None:
    frozen = json.loads(HASHES.read_text())
    local = {}
    for p in source_json or []:
        local.update({d["id"]: d["text"] for d in json.loads(Path(p).read_text())})
    for c in cases:
        status, text = ("local", local[c["id"]]) if c["id"] in local else fetch_text(c)
        items = ordered_entries(text)
        ev = items[c["start"]:c["cut"]]
        evidence = "\n".join(ev)
        c["fetch_status"] = status
        c["timeline_match"] = timeline_hash(items) == c["timeline_sha256"]
        c["evidence_match"] = hashlib.sha256(evidence.encode()).hexdigest() == frozen[c["id"]]["evidence_sha256"]
        c["evidence"] = evidence
        c["n_entries"] = len(ev)


# ---------------------------------------------------------------- AAF side
def blank_evidence() -> dict[str, Any]:
    ev: dict[str, Any] = {}
    for f in FIELD_SPECS:
        b, n = split(f)
        blk = ev.setdefault(b, {})
        blk[n] = None
        blk.setdefault("_evidence", {})[n] = {"status": "missing"}
    return ev


def pm_case(c: dict[str, Any]) -> dict[str, Any]:
    return {"structured": blank_evidence(), "documents": [{"id": DOC_ID, "kind": "incident_timeline", "text": c["evidence"]}], "scope": SCOPE}


def decide(ev: dict[str, Any]) -> str:
    return str(deterministic_assessment(copy.deepcopy(ev))["selected_action"])


def check_ext(x: dict[str, Any], doc: str) -> tuple[bool, str]:
    if x.get("field") not in EXT_SPECS:
        return False, "unknown_field"
    if not isinstance(x.get("value"), bool):
        return False, "type_mismatch"
    if x.get("method", "literal") != "literal":
        return False, "method_not_allowed_for_bool"
    if x.get("doc_id") != DOC_ID:
        return False, "unknown_doc"
    q = x.get("quote") or ""
    if not q.strip() or _norm(q) not in _norm(doc):
        return False, "quote_not_in_document"
    bad = _context_reason(q, doc, SCOPE, VERSION)
    return (False, bad) if bad else (True, "ok")


def aaf_ext(case: dict[str, Any], extractions: list[dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    """L+: numeric/boolean schema fields as in L, then categorical fields mapped into still-missing SRE
    fields, then the one added rule (deploy.change_preceded_impact + material SRE outcome -> Rollback,
    unless a hard-override interaction is active)."""
    core = [x for x in extractions if x.get("field") in FIELD_SPECS]
    ext = [x for x in extractions if x.get("field") in EXT_SPECS]
    ev, info = fill(case, core, VERSION)
    doc = case["documents"][0]["text"]
    cats, ext_rej = {}, []
    for x in ext:
        ok, why = check_ext(x, doc)
        if ok:
            cats[x["field"]] = cats.get(x["field"], False) or bool(x["value"])
        else:
            ext_rej.append({"field": x.get("field"), "reason": why})
    mapped = {}
    for f, (target, val) in EXT_MAP.items():
        if cats.get(f) and target not in info["accepted"]:
            b, n = split(target)
            ev[b][n] = val
            ev[b]["_evidence"][n] = {"status": "measured", "source": f"categorical:{f}"}
            mapped[target] = val
    action = decide(ev)
    rule = False
    if cats.get("deploy.change_preceded_impact") and _material_sre_outcome(ev):
        hard = any(int(i.get("arbitration_tier", 0)) >= TIER_HARD_OVERRIDE for i in interaction_state_v2(ev).get("interactions", []))
        if not hard:
            action, rule = ROLLBACK, True
    return action, {**info, "categorical_accepted": cats, "categorical_rejected": ext_rej, "mapped": mapped, "added_rule_fired": rule}


# ---------------------------------------------------------------- LLM calls
class LLM:
    def __init__(self, mock: bool):
        self.mock = mock
        self.model = os.environ.get("OPENAI_MODEL", "gpt-5.6-luna")
        if not mock:
            from openai import OpenAI
            self.client = OpenAI()

    def call(self, instructions: str, payload: dict[str, Any], effort: str) -> dict[str, Any]:
        t0 = time.perf_counter()
        if self.mock:
            return {"obj": {}, "raw": "mock", "usage": {"total_tokens": 0}, "latency_ms": 0.0, "error": None}
        for attempt in range(4):
            try:
                r = self.client.responses.create(model=self.model, reasoning={"effort": effort}, instructions=instructions,
                                                 input=json.dumps(payload, ensure_ascii=False, sort_keys=True))
                raw = r.output_text or ""
                u = getattr(r, "usage", None)
                usage = {k: int(getattr(u, k, 0) or 0) for k in ("input_tokens", "output_tokens", "total_tokens")}
                return {"obj": _json_obj(raw), "raw": raw, "usage": usage, "latency_ms": (time.perf_counter() - t0) * 1000, "error": None}
            except Exception as exc:  # retried, then recorded as a failed run (never dropped)
                err = f"{type(exc).__name__}: {str(exc)[:200]}"
                time.sleep(5 * (attempt + 1))
        return {"obj": {}, "raw": err, "usage": {"total_tokens": 0}, "latency_ms": (time.perf_counter() - t0) * 1000, "error": err}


def reader_payload(case: dict[str, Any], ext: bool) -> dict[str, Any]:
    p = {"scope": SCOPE, "field_catalogue": FIELD_SPECS,
         "documents": [{"id": d["id"], "kind": d["kind"], "text": d["text"]} for d in case["documents"]]}
    if ext:
        p["categorical_fields"] = EXT_SPECS
    return p


def mock_extractions(case: dict[str, Any], ext: bool) -> list[dict[str, Any]]:
    """Offline only: exercises the pipeline (regex hits as 'LLM' output plus one fabricated quote)."""
    ex = [{"field": f, "value": r["value"], "doc_id": DOC_ID, "quote": r["quote"], "method": "literal"}
          for f, r in regex_reader.read(case["documents"]).items()]
    ex.append({"field": "sec.critical_cves", "value": 7, "doc_id": DOC_ID, "quote": "7 critical CVEs", "method": "literal"})
    if ext:
        first = case["documents"][0]["text"].splitlines()[0][:60]
        ex.append({"field": "sre.outage_present", "value": True, "doc_id": DOC_ID, "quote": first, "method": "literal"})
    return ex


def run_one(llm: LLM, c: dict[str, Any], arm: str, run: int) -> dict[str, Any]:
    case = pm_case(c)
    if arm in ("L", "L+"):
        res = llm.call(EXT_PROMPT if arm == "L+" else READER_SYSTEM_V2, reader_payload(case, arm == "L+"),
                       os.environ.get("OPENAI_REASONING_EFFORT", "none"))
        exs = mock_extractions(case, arm == "L+") if llm.mock else list(res["obj"].get("extractions") or [])
        if arm == "L":
            ev, info = fill(case, [x for x in exs if x.get("field") in FIELD_SPECS], VERSION)
            action = decide(ev)
        else:
            action, info = aaf_ext(case, exs)
    else:  # A
        res = llm.call(A_SYSTEM, {"allowed_actions": ACTIONS, "governance_policy": A_POLICY, "timeline": c["evidence"]},
                       os.environ.get("A_REASONING_EFFORT", "medium"))
        sel = str(res["obj"].get("selected_action", "")) if not llm.mock else MITIGATE
        action = sel if sel in ACTIONS else "INVALID"
        info = {"rationale_summary": res["obj"].get("rationale_summary")}
    return {"case": c["id"], "arm": arm, "run": run, "action": action, "info": info,
            "raw": res["raw"], "usage": res["usage"], "latency_ms": res["latency_ms"], "error": res["error"]}


# ---------------------------------------------------------------- stats
def wilson(k: int, n: int) -> list[float] | None:
    if n == 0:
        return None
    z, p = 1.96, k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [max(0.0, round(100 * (c - h), 1)), min(100.0, round(100 * (c + h), 1))]


def rate(xs: list[bool]) -> dict[str, Any]:
    k, n = sum(xs), len(xs)
    return {"k": k, "n": n, "pct": round(100 * k / n, 1) if n else None, "wilson95": wilson(k, n)}


def mcnemar(a: list[bool], b: list[bool]) -> dict[str, Any]:
    n10 = sum(1 for x, y in zip(a, b) if x and not y)
    n01 = sum(1 for x, y in zip(a, b) if y and not x)
    p = float(binomtest(min(n10, n01), n10 + n01, 0.5).pvalue) if n10 + n01 else 1.0
    return {"first_only": n10, "second_only": n01, "exact_p": float(f"{p:.4g}")}


def holm(ps: dict[str, float]) -> dict[str, float]:
    order = sorted(ps, key=ps.get)
    out, running = {}, 0.0
    for i, k in enumerate(order):
        running = max(running, min(1.0, (len(order) - i) * ps[k]))
        out[k] = float(f"{running:.4g}")
    return out


def encrypt(plain: Path, out_prefix: Path) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        key = Path(tmp) / "k.bin"
        key.write_bytes(os.urandom(32))
        subprocess.run(["openssl", "enc", "-aes-256-cbc", "-pbkdf2", "-salt", "-in", str(plain),
                        "-out", f"{out_prefix}.enc", "-pass", f"file:{key}"], check=True)
        subprocess.run(["openssl", "pkeyutl", "-encrypt", "-pubin", "-inkey", str(PUBKEY), "-in", str(key),
                        "-out", f"{out_prefix}.key.enc"], check=True)
    plain.unlink()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--source-json", nargs="*", help="offline: decrypted bundles (never in CI)")
    ap.add_argument("--out", default=str(HERE / "results"))
    args = ap.parse_args()

    cases = load_cases()
    attach_evidence(cases, args.source_json)
    usable = [c for c in cases if c["evidence_match"]]
    llm = LLM(args.mock)
    jobs = [(c, arm, r) for c in usable for arm in ("L", "A", "L+") for r in range(args.repeats)]
    with ThreadPoolExecutor(max_workers=1 if args.mock else args.workers) as pool:
        runs = list(pool.map(lambda j: run_one(llm, *j), jobs))
    by = defaultdict(list)
    for x in runs:
        by[(x["case"], x["arm"])].append(x)

    rows = []
    for c in cases:
        adm = ADMISSIBLE[c["label"]]
        row = {"case": c["id"], "source": c["source"], "label": c["label"], "n_entries": c["n_entries"],
               "fetch_status": c["fetch_status"], "evidence_match": c["evidence_match"], "timeline_match": c["timeline_match"]}
        if c["evidence_match"]:
            ev_r, info_r = fill(pm_case(c), [{"field": f, "value": v["value"], "doc_id": DOC_ID, "quote": v["quote"], "method": "literal"}
                                              for f, v in regex_reader.read(pm_case(c)["documents"]).items()], VERSION)
            r_act = decide(ev_r)
            row.update({"M0_action": MAJORITY, "M0_correct": MAJORITY in adm, "R_action": r_act, "R_correct": r_act in adm,
                        "R_fields": "|".join(sorted(info_r["accepted"]))})
            for arm in ("L", "A", "L+"):
                rs = by[(c["id"], arm)]
                k = sum(x["action"] in adm for x in rs)
                mode = Counter(x["action"] for x in rs).most_common(1)[0][0]
                row.update({f"{arm}_runs_correct": f"{k}/{len(rs)}", f"{arm}_correct": k * 2 > len(rs), f"{arm}_modal_action": mode,
                            f"{arm}_actions": json.dumps(dict(Counter(x["action"] for x in rs)), sort_keys=True),
                            f"{arm}_errors": sum(1 for x in rs if x["error"])})
                if arm in ("L", "L+"):
                    row[f"{arm}_fields_accepted"] = json.dumps(dict(Counter(f for x in rs for f in x["info"]["accepted"])), sort_keys=True)
                if arm == "L+":
                    row["L+_categorical"] = json.dumps(dict(Counter(f for x in rs for f, v in x["info"]["categorical_accepted"].items() if v)), sort_keys=True)
                    row["L+_rule_fired_runs"] = sum(x["info"]["added_rule_fired"] for x in rs)
        rows.append(row)

    ok = [r for r in rows if r["evidence_match"]]
    arms = ("M0", "R", "L", "A", "L+")
    col = lambda a, rs=ok: [bool(r[f"{a}_correct"]) for r in rs]
    primary = {"L_vs_M0": mcnemar(col("L"), col("M0")), "L_vs_A": mcnemar(col("L"), col("A"))}
    primary["holm_adjusted"] = holm({k: v["exact_p"] for k, v in primary.items()})
    modal = lambda a: dict(Counter(r[f"{a}_modal_action"] if a in ("L", "A", "L+") else r[f"{a}_action"] for r in ok))
    summary = {
        "protocol": "external_validation/postmortems/PROTOCOL.md", "mock": args.mock, "repeats": args.repeats,
        "cases_labelled": len(rows), "cases_evidence_hash_match": len(ok),
        "cases_excluded_hash_mismatch": [r["case"] for r in rows if not r["evidence_match"]],
        "label_distribution": dict(Counter(r["label"] for r in ok)),
        "agreement_with_responders": {a: rate(col(a)) for a in arms},
        "primary": primary,
        "secondary_paired": {"Lplus_vs_L": mcnemar(col("L+"), col("L")), "A_vs_M0": mcnemar(col("A"), col("M0")),
                             "R_vs_M0": mcnemar(col("R"), col("M0")), "Lplus_vs_A": mcnemar(col("L+"), col("A")),
                             "L_vs_R": mcnemar(col("L"), col("R"))},
        "by_label": {lab: {a: rate(col(a, [r for r in ok if r["label"] == lab])) for a in arms} for lab in sorted({r["label"] for r in ok})},
        "by_source": {s: {a: rate(col(a, [r for r in ok if r["source"] == s])) for a in arms} for s in ("danluu", "gitlab")},
        "modal_action_distribution": {a: modal(a) for a in ("R", "L", "A", "L+")},
        "non_decision_rate_modal": {a: rate([m in NON_DECISIONS for m in [r[f"{a}_modal_action"] if a != "R" else r["R_action"] for r in ok]]) for a in ("R", "L", "L+", "A")},
        "run_level_agreement": {a: rate([x["action"] in ADMISSIBLE[next(c["label"] for c in usable if c["id"] == x["case"])] for x in runs if x["arm"] == a]) for a in ("L", "A", "L+")},
        "unanimous_cases": {a: sum(1 for r in ok if len(json.loads(r[f"{a}_actions"])) == 1) for a in ("L", "A", "L+")},
        "llm_failed_calls": sum(1 for x in runs if x["error"]),
        "llm_tokens_total": {a: sum(x["usage"].get("total_tokens", 0) for x in runs if x["arm"] == a) for a in ("L", "A", "L+")},
        "rejections_by_reason": {a: dict(Counter(y["reason"] for x in runs if x["arm"] == a for y in x["info"].get("rejected", []))) for a in ("L", "L+")},
        "Lplus_added_rule_fired_runs": sum(1 for x in runs if x["arm"] == "L+" and x["info"]["added_rule_fired"]),
    }
    meta = {"created_utc": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(),
            "model": None if args.mock else llm.model, "reader_reasoning_effort": os.environ.get("OPENAI_REASONING_EFFORT", "none"),
            "A_reasoning_effort": os.environ.get("A_REASONING_EFFORT", "medium"), "prompt_sha256": prompt_hashes(),
            "labels_sha256": {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in LABEL_FILES},
            "github_run_id": os.environ.get("GITHUB_RUN_ID"), "git_sha": os.environ.get("GITHUB_SHA")}

    out = Path(args.out) / ("mock" if args.mock else datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    (out / "metadata.json").write_text(json.dumps(meta, indent=2))
    with (out / "case_results.csv").open("w", newline="") as fh:
        keys = list(dict.fromkeys(k for r in rows for k in r))
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)
    raw = out / "raw_runs.jsonl"
    with raw.open("w") as fh:  # contains verbatim quotes -> encrypted
        for x in runs:
            fh.write(json.dumps(x, default=str) + "\n")
    encrypt(raw, out / "raw_runs.jsonl")
    print(json.dumps({k: summary[k] for k in ("cases_evidence_hash_match", "agreement_with_responders", "primary")}, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
