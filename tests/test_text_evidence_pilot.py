"""Integrity tests for the text-evidence development pilot (no API calls)."""
from pathlib import Path

import yaml

from text_evidence_pilot.llm_reader import reader_payload
from text_evidence_pilot.run_pilot import fill, missing_in_structured
from text_evidence_pilot.schema import FIELD_SPECS
from text_evidence_pilot.validator import check

CASES = yaml.safe_load((Path(__file__).resolve().parents[1] / "text_evidence_pilot" / "cases_text_pilot_v0.yaml").read_text())["cases"]


def _doc_for(case, quote):
    norm = lambda s: " ".join(s.split())
    return next(d["id"] for d in case["documents"] if norm(quote) in norm(d["text"]))


def test_every_truth_value_passes_the_validator():
    for case in CASES:
        for field, t in (case.get("truth") or {}).items():
            ok, reason = check({"field": field, "doc_id": _doc_for(case, t["quote"]), **t}, case["documents"])
            assert ok, (case["id"], field, reason)


def test_truth_fields_are_exactly_the_missing_fields_or_a_subset():
    for case in CASES:
        assert set(case.get("truth") or {}) <= missing_in_structured(case), case["id"]
        assert set(case.get("truth") or {}) <= set(FIELD_SPECS), case["id"]


def test_fabricated_quote_is_rejected():
    case = CASES[0]
    ok, reason = check({"field": "sec.critical_cves", "value": 3, "doc_id": "DOC-1", "quote": "3 critical CVEs", "method": "literal"}, case["documents"])
    assert not ok and reason == "quote_not_in_document"


def test_real_quote_with_wrong_value_is_rejected():
    case = next(c for c in CASES if c["id"] == "TX-E04")
    ok, reason = check({"field": "sre.error_rate_pct", "value": 25, "doc_id": "DOC-1", "quote": "error rate is currently 11.5%", "method": "literal"}, case["documents"])
    assert not ok and reason == "value_not_supported_by_quote"


def test_readers_cannot_overwrite_structured_telemetry():
    case = {"structured": {"sre": {"p95_latency_ms": 520}},
            "documents": [{"id": "DOC-1", "text": "p95 is 990 ms right now"}]}
    ev, info = fill(case, [{"field": "sre.p95_latency_ms", "value": 990, "doc_id": "DOC-1", "quote": "p95 is 990 ms", "method": "literal"}])
    assert ev["sre"]["p95_latency_ms"] == 520 and not info["accepted"] and info["ignored_present_fields"] == ["sre.p95_latency_ms"]


def test_model_payload_hides_evaluator_fields():
    for case in CASES:
        text = str(reader_payload(case))
        for key in ("truth", "tier", "focus", "structured"):
            assert f"'{key}'" not in text, (case["id"], key)


STRESS = yaml.safe_load((Path(__file__).resolve().parents[1] / "text_evidence_pilot" / "cases_text_stress_v1.yaml").read_text())["cases"]


def test_stress_truth_passes_validator_unless_rejection_is_expected():
    for case in STRESS:
        for field, t in (case.get("truth") or {}).items():
            ok, _ = check({"field": field, "doc_id": _doc_for(case, t["quote"]), **t}, case["documents"])
            assert ok == (case.get("validator_expected") != "reject"), (case["id"], field)


def test_stress_cases_are_decision_sensitive():
    from text_evidence_pilot.run_pilot import ceiling_evidence, decide
    from text_evidence_pilot.schema import split
    for case in STRESS:
        ceiling = decide(ceiling_evidence(case))
        ev = ceiling_evidence(case)
        for f, v in (case.get("decoy") or {}).items():
            b, n = split(f)
            ev.setdefault(b, {})[n] = v
            ev[b].setdefault("_evidence", {})[n] = {"status": "measured"}
        assert ceiling != decide(case["structured"]) or (case.get("decoy") and decide(ev) != ceiling), case["id"]


HOLDOUT = yaml.safe_load((Path(__file__).resolve().parents[1] / "text_evidence_pilot" / "cases_text_holdout_v2.yaml").read_text())["cases"]


def test_v1_checker_on_all_truths_matches_declared_expectations():
    for case in CASES + STRESS + HOLDOUT:
        for field, t in (case.get("truth") or {}).items():
            x = {"field": field, "doc_id": _doc_for(case, t["quote"]), **t}
            if case["id"] == "ST-U2":  # stored with the v0 method; v1 verifies it as a ratio
                x["method"] = "ratio_percent"
            ok, reason = check(x, case["documents"], "v1", case.get("scope"))
            assert ok == (case.get("validator_expected") != "reject_v1"), (case["id"], field, reason)


def test_v1_guard_rejects_non_current_quotes_and_v0_does_not():
    docs = [{"id": "D", "text": "From last month's postmortem: cart scaled to 18 replicas. In staging, error rate hit 14%. "
                                 "If p95 goes above 900 ms, page. Could the error rate be 15%?"}]
    scope = {"service": "cart", "environment": "production"}
    probes = [("finops.hpa_scale_to", 18, "cart scaled to 18 replicas", "historical_context"),
              ("sre.error_rate_pct", 14, "error rate hit 14%", "other_environment"),
              ("sre.p95_latency_ms", 900, "p95 goes above 900 ms", "target_threshold_or_plan"),
              ("sre.error_rate_pct", 15, "the error rate be 15%", "speculative")]
    for field, value, quote, reason in probes:
        x = {"field": field, "value": value, "doc_id": "D", "quote": quote, "method": "literal"}
        assert check(x, docs, "v0", scope)[0]
        assert check(x, docs, "v1", scope) == (False, reason)


def test_ratio_method_only_in_v1():
    docs = [{"id": "D", "text": "120 of 1,000 requests failed"}]
    x = {"field": "sre.error_rate_pct", "value": 12, "doc_id": "D", "quote": "120 of 1,000 requests failed", "method": "ratio_percent"}
    assert check(x, docs, "v1") == (True, "ok") and check(x, docs, "v0")[1] == "unknown_method"


HOLDOUT3 = yaml.safe_load((Path(__file__).resolve().parents[1] / "text_evidence_pilot" / "cases_text_holdout_v3.yaml").read_text())["cases"]


def test_v2_checker_on_all_truths_matches_declared_expectations():
    for case in CASES + STRESS + HOLDOUT + HOLDOUT3:
        results = []
        for field, t in (case.get("truth") or {}).items():
            x = {"field": field, "doc_id": _doc_for(case, t["quote"]), **t}
            if case["id"] == "ST-U2":
                x["method"] = "ratio_percent"
            results.append(check(x, case["documents"], "v2", case.get("scope"))[0])
        if case.get("validator_expected") == "reject_v2":
            assert not all(results), case["id"]  # at least one truth field is known to be unverifiable
        else:
            assert all(results), case["id"]


def test_v2_current_wording_overrides_context_only_in_its_own_subclause():
    d = lambda t: [{"id": "D", "text": t}]
    x = lambda q, f="sre.error_rate_pct", v=13.5: {"field": f, "value": v, "doc_id": "D", "quote": q, "method": "literal"}
    t = "Compared with yesterday's 2%, the auth error rate is now 13.5%."
    assert check(x("the auth error rate is now 13.5%"), d(t), "v1") == (False, "historical_context")
    assert check(x("the auth error rate is now 13.5%"), d(t), "v2") == (True, "ok")
    t2 = "As of last month, ledger spend is up 35% vs baseline."
    assert not check(x("ledger spend is up 35%", "finops.cost_spike_pct", 35), d(t2), "v2")[0]


def test_v2_ratio_either_order():
    docs = [{"id": "D", "text": "out of 400 requests in the window, 36 failed"}]
    x = {"field": "sre.error_rate_pct", "value": 9, "doc_id": "D", "quote": "out of 400 requests in the window, 36 failed", "method": "ratio_percent"}
    assert not check(x, docs, "v1")[0] and check(x, docs, "v2") == (True, "ok")
