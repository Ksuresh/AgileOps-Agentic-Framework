"""Deterministic timeline extraction from postmortem page text.

An entry starts at a line that begins with a time (optionally preceded by a date or "At"),
e.g. "14:43 UTC: ...", "2020-11-02 14:44 UTC: ...", "At 17:57 UTC ...", "15:09" (with the
description on the following line). Following non-timestamp lines are joined into the entry
until the next timestamp line, capped at 600 characters. Entries are numbered from 0; the
study's cut index refers to these numbers. A SHA-256 of the joined entries detects changes
in the source page between screening and run time.
"""
from __future__ import annotations

import hashlib
import re

MONTH = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?"
DATE = rf"(?:\d{{4}}-\d{{2}}-\d{{2}}|\d{{1,2}}/\d{{1,2}}(?:/\d{{2,4}})?|{MONTH}\s+\d{{1,2}}(?:,?\s+\d{{4}})?|\d{{1,2}}\s+{MONTH}(?:\s+\d{{4}})?)"
TIME = r"\d{1,2}:\d{2}(?::\d{2})?(?:\s*(?:am|pm))?(?:\s*(?:utc|gmt|pt|pst|pdt|et|est|edt|bst|cet|cest|z|us/pacific))?"
START = re.compile(rf"^\W{{0,3}}(?:(?:on\s+)?{DATE}[,\s]+(?:at\s+)?)?(?:at\s+|by\s+|around\s+|approximately\s+)?{TIME}\b", re.I)
MAX_ENTRY = 600


def entries(text: str) -> list[str]:
    out: list[str] = []
    cur: list[str] | None = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if START.match(line):
            if cur:
                out.append(" ".join(cur)[:MAX_ENTRY])
            cur = [line]
        elif cur is not None:
            if sum(len(x) for x in cur) < MAX_ENTRY:
                cur.append(line)
    if cur:
        out.append(" ".join(cur)[:MAX_ENTRY])
    return out


HHMM = re.compile(r"\b(\d{1,2}):(\d{2})\b")


def _minutes(entry: str) -> int | None:
    m = HHMM.search(entry)
    return None if not m else int(m.group(1)) * 60 + int(m.group(2))


def ordered_entries(text: str) -> list[str]:
    """entries() in chronological order: status pages list newest first, so a list whose
    consecutive leading times mostly decrease (>= 70% of comparable pairs) is reversed."""
    items = entries(text)
    mins = [_minutes(e) for e in items]
    pairs = [(a, b) for a, b in zip(mins, mins[1:]) if a is not None and b is not None and a != b]
    if pairs and sum(1 for a, b in pairs if b < a) / len(pairs) >= 0.7:
        items = items[::-1]
    return items


def timeline_hash(items: list[str]) -> str:
    return hashlib.sha256("\n".join(items).encode()).hexdigest()
