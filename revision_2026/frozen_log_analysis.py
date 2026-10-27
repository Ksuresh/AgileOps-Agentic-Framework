"""Reviewer items answered from the FROZEN manuscript artifacts (no new runs):
R4-3 / R6-3 per-stratum table (agreement, LLM calls, tool requests, schema / evidence-reference failures, tokens,
latency); R4-4 the non-beneficial governance overrides; R2-5 / R1-6 calls, tools and latency per case."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "paper_results" / "frozen_artifacts"
OUT = Path(__file__).resolve().parent / "results"
STUDIES = {"primary": ("agentic_primary_34428945881", "cases_v1.yaml"),
           "replication": ("agentic_replication_34432645600", "cases_replication_v1.yaml")}


def main() -> None:
    OUT.mkdir(exist_ok=True)
    rows, overrides = [], []
    for study, (d, cf) in STUDIES.items():
        raw = {x["case_id"]: x for x in json.loads((ART / d / "raw_outputs.json").read_text())}
        cases = {c["id"]: c for c in yaml.safe_load((ROOT / "agentic_experiments" / cf).read_text())["cases"]}
        for r in csv.DictReader((ART / d / "case_results.csv").open()):
            x = raw[r["case_id"]]
            ao, hy = x["agentic_only"], x["hybrid"]
            b = lambda v: str(v) == "True"
            row = {"study": study, "case": r["case_id"], "stratum": r["stratum"],
                   "det_ok": b(r["deterministic_correct"]), "ao_ok": b(r["agentic_only_correct"]), "hy_ok": b(r["hybrid_correct"]),
                   "hy_invoked": b(r["hybrid_invoked"]),
                   "ao_llm_calls": len(ao.get("call_audit") or []), "hy_llm_calls": len(hy.get("call_audit") or []),
                   "ao_tool_requests": len(ao.get("tool_events") or []), "hy_tool_requests": len(hy.get("tool_events") or []),
                   "ao_schema_fail": not b(r["agentic_only_schema_valid"]), "hy_schema_fail": not b(r["hybrid_schema_valid"]),
                   "ao_evref_fail": not b(r["agentic_only_refs_valid"]), "hy_evref_fail": not b(r["hybrid_refs_valid"]),
                   "ao_tokens": int(r["agentic_only_tokens"]), "hy_tokens": int(r["hybrid_tokens"]),
                   "ao_latency_s": float(r["agentic_only_latency_ms"]) / 1000, "hy_latency_s": float(r["hybrid_latency_ms"]) / 1000,
                   "override": b(r["governance_override"]), "beneficial": b(r["beneficial_override"])}
            rows.append(row)
            if row["override"] and not row["beneficial"]:
                adm = cases[r["case_id"]]["admissible_actions"]
                overrides.append({"study": study, "case": r["case_id"], "stratum": r["stratum"], "focus": cases[r["case_id"]].get("focus", ""),
                                  "llm_proposal": hy.get("agent_proposal"), "proposal_ok": hy.get("agent_proposal") in adm,
                                  "final_action": hy["selected_action"], "final_ok": hy["selected_action"] in adm, "admissible": adm,
                                  "interaction": ((hy.get("governance") or {}).get("interaction_state") or {}).get("dominant_interaction", {}) and
                                  ((hy.get("governance") or {}).get("interaction_state") or {}).get("dominant_interaction", {}).get("name"),
                                  "tools": [t["tool"] for t in hy.get("tool_events") or []]})
    with (OUT / "frozen_per_case.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    agg = defaultdict(lambda: defaultdict(float))
    for r in rows:
        for key in (f"{r['study']}|{r['stratum']}", f"{r['study']}|ALL", f"combined|{r['stratum']}", "combined|ALL"):
            a = agg[key]; a["n"] += 1
            for k, v in r.items():
                if isinstance(v, (bool, int, float)) and k not in ("study",):
                    a[k] += float(v)
    table = {k: {kk: (round(vv, 1) if not kk.endswith("latency_s") else round(vv / v["n"], 2)) for kk, vv in v.items()} for k, v in agg.items()}
    for k, v in table.items():  # means for latency are per case; also report invoked-only hybrid latency
        inv = [r["hy_latency_s"] for r in rows if r["hy_invoked"] and (k == "combined|ALL" or k.split("|")[0] in (r["study"], "combined") and k.split("|")[1] in (r["stratum"], "ALL"))]
        v["hy_latency_s_mean_when_invoked"] = round(sum(inv) / len(inv), 2) if inv else 0.0
    (OUT / "frozen_per_stratum.json").write_text(json.dumps(table, indent=1))
    (OUT / "nonbeneficial_overrides.json").write_text(json.dumps(overrides, indent=1))
    print(json.dumps(table["combined|ALL"], indent=0)); print(json.dumps(overrides, indent=1))
    for s in ("CLEAR", "AMBIGUOUS", "INCOMPLETE", "MISLEADING"):
        t = table[f"combined|{s}"]
        print(s, {k: t[k] for k in ("n", "det_ok", "ao_ok", "hy_ok", "hy_invoked", "ao_llm_calls", "hy_llm_calls", "hy_tool_requests", "ao_schema_fail", "hy_schema_fail", "ao_evref_fail", "hy_evref_fail", "override", "beneficial")})


if __name__ == "__main__":
    main()
