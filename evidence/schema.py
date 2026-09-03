from __future__ import annotations

"""Evidence-status helpers for controlled and runtime experiments.

The framework distinguishes four states for each evidence field:
- measured: directly observed from the source represented by the field
- proxy: derived from a different observable and explicitly labelled as such
- missing: expected/relevant but unavailable
- not_applicable: field is not meaningful for the case/source

Legacy synthetic scenarios that do not provide metadata remain supported. Their
explicit values are treated as `measured` because they are controlled injected
experimental values, not runtime measurements.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional

VALID_STATUSES = {"measured", "proxy", "missing", "not_applicable"}


@dataclass(frozen=True)
class EvidenceValue:
    value: Any
    status: str
    source: str = ""
    note: str = ""

    @property
    def usable(self) -> bool:
        return self.status in {"measured", "proxy"} and self.value is not None

    @property
    def is_proxy(self) -> bool:
        return self.status == "proxy"


def _meta(block: Dict[str, Any], field: str) -> Dict[str, Any]:
    metadata = block.get("_evidence", {}) or {}
    item = metadata.get(field, {}) or {}
    return item if isinstance(item, dict) else {}


def get_evidence(block: Dict[str, Any], field: str, default: Any = None) -> EvidenceValue:
    item = _meta(block, field)
    explicit_status = item.get("status")

    if explicit_status is not None:
        status = str(explicit_status).strip().lower()
        if status not in VALID_STATUSES:
            raise ValueError(f"Invalid evidence status for {field}: {status}")
        return EvidenceValue(
            value=block.get(field, default),
            status=status,
            source=str(item.get("source", "")),
            note=str(item.get("note", "")),
        )

    # Backward compatibility for existing controlled scenarios.
    if block.get("_missing") is True:
        return EvidenceValue(default, "missing", source="legacy domain-level missing marker")
    if field in block:
        return EvidenceValue(block.get(field), "measured", source="controlled/legacy explicit value")
    return EvidenceValue(default, "missing", source="field absent")


def numeric_evidence(block: Dict[str, Any], field: str) -> EvidenceValue:
    ev = get_evidence(block, field, default=None)
    if not ev.usable:
        return ev
    try:
        return EvidenceValue(float(ev.value), ev.status, ev.source, ev.note)
    except (TypeError, ValueError):
        return EvidenceValue(None, "missing", ev.source, f"Non-numeric value: {ev.value!r}")


def bool_evidence(block: Dict[str, Any], field: str) -> EvidenceValue:
    ev = get_evidence(block, field, default=None)
    if not ev.usable:
        return ev
    return EvidenceValue(bool(ev.value), ev.status, ev.source, ev.note)


def provenance_suffix(ev: EvidenceValue) -> str:
    if ev.status == "proxy":
        return " [proxy]"
    return ""


def usable_count(block: Dict[str, Any], fields: list[str]) -> int:
    return sum(1 for field in fields if get_evidence(block, field).usable)


def missing_count(block: Dict[str, Any], fields: list[str]) -> int:
    return sum(1 for field in fields if get_evidence(block, field).status == "missing")
