"""Audit UCI incident management dataset metadata without downloading incident records."""
from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

URL = "https://archive.ics.uci.edu/api/dataset/498"
OUT = Path(__file__).resolve().parent / "results/external_dataset_audit/uci_incident_metadata.json"

def main():
    with urlopen(Request(URL, headers={"User-Agent": "ieee-access-dataset-audit/1.0"}), timeout=60) as r:
        data = json.load(r)
    metadata = data.get("metadata") or {}
    variables = data.get("variables") or []
    report = {
        "status": "source_metadata_only_not_incident_record_inspection",
        "retrieved_utc": datetime.now(timezone.utc).isoformat(),
        "source": URL,
        "name": metadata.get("name"),
        "doi": metadata.get("doi"),
        "license": metadata.get("license"),
        "num_instances": metadata.get("num_instances"),
        "num_features": metadata.get("num_features"),
        "variable_count": len(variables),
        "variables": [
            {"name": v.get("name"), "role": v.get("role"), "type": v.get("type"),
             "missing_values": v.get("missing_values")}
            for v in variables
        ],
        "data_url": (data.get("data") or {}).get("data_url"),
        "limitation": "Public metadata is not a validated governance decision label or an incident-level sample.",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
