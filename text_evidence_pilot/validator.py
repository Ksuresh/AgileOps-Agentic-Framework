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

Version v1 (2026-10-21, designed after the stress-test v1 results; evaluate on the fresh
holdout set, not on the cases that motivated it) adds:
  ratio_percent      value == 100 * a / b for the first two numbers in the quote ("120 of 1,000").
  context guard      reject a value whose clause (text from the clause start up to the end of the
                     quote, plus a trailing "?") marks it as historical, speculative, a
                     target/threshold/budget/plan, or from an environment other than scope.

Version v2 (2026-10-21, designed after the holdout-v2 results; evaluate on holdout v3):
  context guard      as v1, except that a past/conditional/speculative phrase elsewhere in the
                     sentence is ignored when the value's own sub-clause (after the last comma or
                     "but/while/whereas") carries explicit current wording (current, now, today,
                     as of, latest) and itself contains no such phrase.
  ratio_percent      accepts A/B or B/A (first two numbers), provided the value is <= 100.
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
METHODS_V1 = METHODS | {"ratio_percent"}
HIST = re.compile(r"\b(?:last|previous|prior|earlier)\s+(?:week|month|quarter|year|incident|outage|release|sprint|deploy(?:ment)?)\b"
                  r"|\byesterday\b|\bpost-?mortem\b|\b(?:\d+|a|one|two|three|few)\s+(?:days?|weeks?|months?|years?)\s+ago\b", re.I)
NONMEAS = re.compile(r"\b(?:target|threshold|budget|limit|forecast|projected|planned|plan to|will|if|alert when)\b", re.I)
SPEC = re.compile(r"\b(?:could|might|maybe|perhaps|not sure|i think|guess|rumou?r|possibly)\b", re.I)
ENVS = {"production", "prod", "staging", "stage", "sandbox", "dev", "development", "test", "qa", "preprod", "uat"}
BOUNDARY = re.compile(r"(?<!\d)[.;!?](?!\d)|\n")
SUB_BOUNDARY = re.compile(r"(?<!\d),(?!\d)|\b(?:but|while|whereas)\b", re.I)
CURRENT = re.compile(r"\b(?:current|currently|now|today|as of|latest|at the moment)\b", re.I)
GUARDED = ("v1", "v2")


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


def _time_reason(clause: str, tail: str) -> str | None:
    if HIST.search(clause):
        return "historical_context"
    if NONMEAS.search(clause):
        return "target_threshold_or_plan"
    if SPEC.search(clause) or "?" in tail:
        return "speculative"
    return None


def _context_reason(quote: str, doc_text: str, scope: dict[str, Any] | None, version: str = "v1") -> str | None:
    if "\n" in quote.strip():  # multi-line log blocks are checked on the quote itself
        clause, sub, tail = quote, quote, ""
    else:
        text, q = _norm(doc_text), _norm(quote)
        i = text.find(q)
        starts = [m.end() for m in BOUNDARY.finditer(text, 0, i)]
        c0 = starts[-1] if starts else 0
        clause = text[c0: i + len(q)]
        subs = [m.end() for m in SUB_BOUNDARY.finditer(text, c0, i)]
        sub = text[(subs[-1] if subs else c0): i + len(q)]
        m = BOUNDARY.search(text, i + len(q))
        tail = text[i + len(q): m.end() if m else len(text)]
    reason = _time_reason(clause, tail)
    if reason and version == "v2" and CURRENT.search(sub) and not _time_reason(sub, ""):
        reason = None  # explicit current wording attached to the value wins over context elsewhere
    if reason:
        return reason
    env = str((scope or {}).get("environment", "")).lower()
    if env:
        allowed = {env} | ({"prod", "production"} if env in {"prod", "production"} else set())
        words = set(re.findall(r"[a-z]+", clause.lower()))
        if (words & ENVS) - allowed and not words & allowed:
            return "other_environment"
    return None


def check(extraction: dict[str, Any], documents: list[dict[str, str]], version: str = "v0",
          scope: dict[str, Any] | None = None) -> tuple[bool, str]:
    ok, reason = _check_core(extraction, documents, version)
    if ok and version in GUARDED:
        doc = next(d["text"] for d in documents if d["id"] == extraction["doc_id"])
        bad = _context_reason(extraction["quote"], doc, scope, version)
        if bad:
            return False, bad
    return ok, reason


def _check_core(extraction: dict[str, Any], documents: list[dict[str, str]], version: str) -> tuple[bool, str]:
    field = extraction.get("field")
    if field not in FIELD_SPECS:
        return False, "unknown_field"
    spec = FIELD_SPECS[field]
    value = extraction.get("value")
    method = extraction.get("method", "literal")
    if method not in (METHODS_V1 if version in GUARDED else METHODS):
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
    if method == "ratio_percent":
        nums = [float(x.replace(",", "")) for x in re.findall(r"\d[\d,]*(?:\.\d+)?", quote)]
        if len(nums) < 2 or nums[1] == 0:
            return False, "ratio_needs_two_numbers"
        ratios = [100.0 * nums[0] / nums[1]]
        if version == "v2" and nums[0] != 0:
            ratios.append(100.0 * nums[1] / nums[0])
        ok = any(r <= 100.0 and _close(r, value) for r in ratios)
        return (True, "ok") if ok else (False, "ratio_mismatch")
    for n in numbers_in(quote):
        if any(_close(n * f, value) for f in (1.0, 60.0, 1000.0)):
            return True, "ok"
    return False, "value_not_supported_by_quote"
