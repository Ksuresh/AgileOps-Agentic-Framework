"""T1: no-LLM text reader (keyword + regex). Written from the field schema BEFORE the
pilot cases were authored, to keep it a fair, generic baseline rather than one tuned
to (or against) specific cases.

Policy choices a reasonable engineer would make:
- synonyms and common unit spellings per field (ms / s, %, number words);
- a short negation window ("no", "not", "without", "0 ") before boolean cues;
- when a field is mentioned several times, keep the LAST mention, because
  documents are supplied in chronological order and later readings supersede
  earlier ones.
"""
from __future__ import annotations

import re
from typing import Any

WORDS = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
    "fifteen sixteen seventeen eighteen nineteen twenty".split())}
NUM = r"(\d+(?:\.\d+)?|" + "|".join(WORDS) + r")"
GAP = r"[^.\n;]{0,40}?"  # stay inside one clause/line
# Amendment A1 (2026-10-21, after first mock run): removed bare "0" from the negation list;
# it fired on version strings such as "3.2.0". Logged in AMENDMENTS.md.
NEG = re.compile(r"\b(no|not|without|zero|never|none|passed|clean)\b[^.\n;]{0,25}$", re.I)


def _num(tok: str) -> float:
    tok = tok.lower()
    return float(WORDS[tok]) if tok in WORDS else float(tok)


NUMERIC: dict[str, list[tuple[str, Any]]] = {
    "sre.p95_latency_ms": [
        (r"\b(?:p95|p-95|95th[ -]percentile)(?: latency)?" + GAP + NUM + r"\s*(ms|milliseconds|s|sec|seconds)\b",
         lambda m: _num(m.group(1)) * (1000.0 if m.group(2).lower() in {"s", "sec", "seconds"} else 1.0)),
    ],
    "sre.error_rate_pct": [
        (r"\b(?:error[ -]rate|5xx(?: rate)?|errors?)" + GAP + NUM + r"\s*(?:%|percent)", lambda m: _num(m.group(1))),
    ],
    "sre.saturation_pct": [
        (r"\b(?:saturation|utili[sz]ation|cpu usage|cpu at)" + GAP + NUM + r"\s*(?:%|percent)", lambda m: _num(m.group(1))),
    ],
    "sre.availability_pct": [
        (r"\b(?:availability|uptime|slo attainment)" + GAP + r"(\d{2,3}(?:\.\d+)?)\s*(?:%|percent)", lambda m: float(m.group(1))),
    ],
    "finops.cost_spike_pct": [
        (r"\b(?:cost|spend|bill|billing)" + GAP + NUM + r"\s*(?:%|percent)", lambda m: _num(m.group(1))),
    ],
    "finops.hpa_scale_to": [
        (r"\b(?:scaled|scale[sd]? (?:out|up)|hpa)" + GAP + r"\bto\s+" + NUM + r"\s*(?:pods|replicas)?", lambda m: _num(m.group(1))),
        (r"\b" + NUM + r"\s+(?:pods|replicas)\b", lambda m: _num(m.group(1))),
    ],
    "finops.cpu_request_increase_pct": [
        (r"\bcpu requests?" + GAP + NUM + r"\s*(?:%|percent)", lambda m: _num(m.group(1))),
    ],
    "finops.memory_request_increase_pct": [
        (r"\b(?:memory|mem) requests?" + GAP + NUM + r"\s*(?:%|percent)", lambda m: _num(m.group(1))),
    ],
    "sec.critical_cves": [
        (r"\b" + NUM + r"\s+critical\s+(?:cves?|vulnerabilit(?:y|ies))", lambda m: _num(m.group(1))),
        (r"\bcritical\s+(?:cves?|vulnerabilit(?:y|ies))\s*[:=]\s*" + NUM, lambda m: _num(m.group(1))),
    ],
    "deploy.restart_burst_count": [
        (r"\b" + NUM + r"\s+(?:pod |container )?restarts?\b", lambda m: _num(m.group(1))),
        (r"\brestarted\s+" + NUM + r"\s+times\b", lambda m: _num(m.group(1))),
    ],
    "deploy.restart_window_seconds": [
        (r"\b(?:within|in|over)\s+" + NUM + r"\s*(s|sec|secs|seconds|min|mins|minutes?)\b",
         lambda m: _num(m.group(1)) * (60.0 if m.group(2).lower().startswith("min") else 1.0)),
    ],
}

BOOLEAN: dict[str, list[str]] = {
    "deploy.pipeline_failed": [r"\bpipeline\b" + GAP + r"\b(?:failed|failure|red|broken)\b", r"\bbuild (?:failed|failure)\b"],
    "deploy.config_drift": [r"\b(?:config(?:uration)?)\s+drift\b", r"\bdrift(?:ed)? from (?:the )?(?:approved |baseline )?config"],
    "deploy.artifact_mismatch": [r"\b(?:artifact|image|digest)\b" + GAP + r"\bmismatch", r"\bdigest does not match\b"],
    "sec.policy_violation": [r"\bpolicy(?: gate| check)?\b" + GAP + r"\b(?:violation|violated|failed|blocked)\b"],
    "sec.iam_drift": [r"\biam\b" + GAP + r"\bdrift"],
}


def read(documents: list[dict[str, str]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for doc in documents:  # chronological order; later mentions overwrite earlier ones
        text = doc["text"]
        for field, patterns in NUMERIC.items():
            for pattern, conv in patterns:
                for m in re.finditer(pattern, text, re.I):
                    try:
                        out[field] = {"value": conv(m), "doc_id": doc["id"], "quote": m.group(0)}
                    except (KeyError, ValueError):
                        continue
        for field, patterns in BOOLEAN.items():
            for pattern in patterns:
                for m in re.finditer(pattern, text, re.I):
                    before = text[max(0, m.start() - 30):m.start()] + m.group(0)
                    value = not bool(NEG.search(before.split(m.group(0))[0])) and not re.search(r"\bno\b|\bnot\b", m.group(0), re.I)
                    out[field] = {"value": value, "doc_id": doc["id"], "quote": m.group(0)}
    return out
