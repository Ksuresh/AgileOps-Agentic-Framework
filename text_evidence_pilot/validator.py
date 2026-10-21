"""Deterministic check of every reader extraction (regex or LLM) before the rules may use it.

An extraction is accepted only if:
  1. the field is in FIELD_SPECS and the value has the right type;
  2. doc_id names a supplied document and quote is an exact substring of it
     (whitespace-normalised);
  3. the value follows from the quote by the declared method, recomputed here:
       literal            a number in the quote (digits, number words incl. compounds such
                          as "thirty-eight", or "no/none/zero" for 0), optionally scaled by a
                          unit factor (x60 minutes->s, x1000 s->ms), matches the value;
                          for booleans the quote must be non-empty;
       count_lines        value == number of non-empty lines in the quote;
       time_span_seconds  value == seconds between first and last HH:MM:SS in the quote.
The validator never decides actions; it only gates what evidence reaches the rules.
"""
from __future__ import annotations

import re
from typing import Any

from text_evidence_pilot.schema import FIELD_SPECS

ONES = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
    "fifteen sixteen seventeen eighteen nineteen".split())}
TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}
METHODS = {"literal", "count_lines", "time_span_seconds"}


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def numbers_in(text: str) -> list[float]:
    t = text.lower()
    out = [float(x.replace(",", "")) for x in re.findall(r"\d[\d,]*(?:\.\d+)?", t)]
    for m in re.finditer(r"\b(" + "|".join(TENS) + r")(?:[- ](" + "|".join(ONES) + r"))?\b", t):
        out.append(float(TENS[m.group(1)] + (ONES[m.group(2)] if m.group(2) else 0)))
    for w, v in ONES.items():
        if re.search(r"\b" + w + r"\b", t):
            out.append(float(v))
    if re.search(r"\b(no|none|zero)\b", t):
        out.append(0.0)
    return out


def _close(a: float, b: float) -> bool:
    return abs(a - b) <= max(1e-6, 0.01 * abs(b))


def _seconds(ts: str) -> int:
    h, m, s = (int(x) for x in ts.split(":"))
    return h * 3600 + m * 60 + s


def check(extraction: dict[str, Any], documents: list[dict[str, str]]) -> tuple[bool, str]:
    field = extraction.get("field")
    if field not in FIELD_SPECS:
        return False, "unknown_field"
    spec = FIELD_SPECS[field]
    value = extraction.get("value")
    method = extraction.get("method", "literal")
    if method not in METHODS:
        return False, "unknown_method"
    docs = {d["id"]: d["text"] for d in documents}
    doc_id, quote = extraction.get("doc_id"), extraction.get("quote") or ""
    if doc_id not in docs:
        return False, "unknown_doc"
    if not quote.strip() or _norm(quote) not in _norm(docs[doc_id]):
        return False, "quote_not_in_document"
    if spec["type"] == "bool":
        if not isinstance(value, bool):
            return False, "type_mismatch"
        return (True, "ok") if method == "literal" else (False, "method_not_allowed_for_bool")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False, "type_mismatch"
    value = float(value)
    if method == "count_lines":
        n = len([ln for ln in quote.splitlines() if ln.strip()])
        return (True, "ok") if _close(n, value) else (False, "count_mismatch")
    if method == "time_span_seconds":
        stamps = re.findall(r"\b\d{2}:\d{2}:\d{2}\b", quote)
        if len(stamps) < 2:
            return False, "no_timestamps"
        span = _seconds(stamps[-1]) - _seconds(stamps[0])
        return (True, "ok") if _close(span, value) else (False, "span_mismatch")
    for n in numbers_in(quote):
        if any(_close(n * f, value) for f in (1.0, 60.0, 1000.0)):
            return True, "ok"
    return False, "value_not_supported_by_quote"
