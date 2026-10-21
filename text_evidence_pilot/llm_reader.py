"""T2: LLM text reader. The model only proposes field values with a verbatim quote;
it never sees or proposes governance actions. Every proposal is re-checked by
validator.check before the deterministic AAF rules may use it.
"""
from __future__ import annotations

import json
import os
import re
import time
from typing import Any

from text_evidence_pilot.schema import FIELD_SPECS

READER_SYSTEM = """You extract operational evidence values from incident documents for a governance experiment.
You do NOT recommend or decide any action.

Rules:
- Only extract fields listed in field_catalogue, and only for the service named in scope. Ignore other services.
- Extract the CURRENT value at the time of the latest document. If a value was later superseded or fixed, use the latest one.
- Only extract a field when the documents actually state or directly show it. If unsure or absent, omit the field. Never guess.
- For every extraction give doc_id and quote: an exact, verbatim, contiguous substring of that document that supports the value.
- method must be one of:
    "literal"            the value is stated in the quote (digits or number words; "no"/"none" means 0). Unit conversion
                         to the catalogue unit is allowed (e.g. 0.9 s -> 900 ms, one minute -> 60 seconds).
    "count_lines"        the value is the number of lines in a quoted multi-line log block (quote exactly those lines).
    "time_span_seconds"  the value is the seconds between the first and last HH:MM:SS timestamps in the quoted block.
- Values: numbers for number fields (in the catalogue unit), true/false for boolean fields.
Return only strict JSON: {"extractions": [{"field": ..., "value": ..., "doc_id": ..., "quote": ..., "method": ...}]}
"""


def _json_obj(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, re.S)
        return json.loads(m.group(0)) if m else {"extractions": []}


def reader_payload(case: dict[str, Any]) -> dict[str, Any]:
    """Model-visible input. Evaluator-only keys (truth, tier, focus, structured values) are not included."""
    return {
        "scope": case["scope"],
        "field_catalogue": {k: {kk: vv for kk, vv in v.items()} for k, v in FIELD_SPECS.items()},
        "documents": [{"id": d["id"], "kind": d.get("kind", ""), "text": d["text"]} for d in case["documents"]],
    }


class OpenAIReader:
    def __init__(self, model: str | None = None, reasoning_effort: str | None = None):
        from openai import OpenAI
        self.client = OpenAI()
        self.model = model or os.environ.get("OPENAI_MODEL", "gpt-5.6-luna")
        self.reasoning_effort = reasoning_effort or os.environ.get("OPENAI_REASONING_EFFORT", "none")

    def read(self, case: dict[str, Any]) -> dict[str, Any]:
        t0 = time.perf_counter()
        try:
            r = self.client.responses.create(
                model=self.model,
                reasoning={"effort": self.reasoning_effort},
                instructions=READER_SYSTEM,
                input=json.dumps(reader_payload(case), ensure_ascii=False, sort_keys=True),
            )
            raw = r.output_text or ""
            u = getattr(r, "usage", None)
            usage = {k: int(getattr(u, k, 0) or 0) for k in ("input_tokens", "output_tokens", "total_tokens")}
            obj, error = _json_obj(raw), None
        except Exception as exc:  # recorded as a failed run, never silently dropped
            raw, usage, obj, error = str(exc), {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}, {"extractions": []}, type(exc).__name__
        return {"extractions": list(obj.get("extractions") or []), "raw": raw, "usage": usage,
                "latency_ms": (time.perf_counter() - t0) * 1000.0, "error": error}


class MockReader:
    """Offline stand-in for dry runs and CI tests: returns the ground truth plus one
    fabricated extraction per case so the validator's rejection path is exercised.
    Its scores are meaningless as results."""

    def read(self, case: dict[str, Any]) -> dict[str, Any]:
        ex = [{"field": f, "doc_id": _doc_for(case, t["quote"]), **t} for f, t in (case.get("truth") or {}).items()]
        ex.append({"field": "sec.critical_cves", "value": 7, "doc_id": case["documents"][0]["id"],
                   "quote": "7 critical CVEs in production", "method": "literal"})
        return {"extractions": ex, "raw": "mock", "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                "latency_ms": 0.0, "error": None}


def _doc_for(case: dict[str, Any], quote: str) -> str:
    norm = lambda s: re.sub(r"\s+", " ", s).strip()
    for d in case["documents"]:
        if norm(quote) in norm(d["text"]):
            return d["id"]
    return case["documents"][0]["id"]
