"""Audit UCI Incident Management Process Enriched Event Log using official repository metadata.

UCI website and API routes are not assumed stable. A missing API route is
recorded as an acquisition blocker rather than failing the RCAEval audit.
"""
from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

URLS = [
    "https://archive.ics.uci.edu/api/datasets/498",
    "https://archive.ics.uci.edu/dataset/498/incident+management+process+enriched+event+log",
]
OUT = Path(__file__).resolve().parent / "results/external_dataset_audit/uci_incident_metadata.json"

def main():
    attempts = []
    report = {
        "status": "source_access_unverified",
        "retrieved_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_id": 498,
        "source_attempts": attempts,
        "limitation": "Metadata is not incident-level telemetry or governance ground truth.",
    }
    for url in URLS:
        try:
            with urlopen(Request(url, headers={"User-Agent": "Mozilla/5.0 research-metadata-audit"}), timeout=45) as response:
                raw = response.read(2_000_001)
                if len(raw) > 2_000_000:
                    raise ValueError("Metadata response exceeds size limit")
                content_type = response.headers.get("Content-Type", "")
            attempts.append({"url": url, "status": "reachable", "content_type": content_type})
            if "json" in content_type.lower():
                data = json.loads(raw)
                if not isinstance(data, dict):
                    raise ValueError("Unexpected UCI JSON metadata")
                metadata = data.get("metadata") or data
                if isinstance(metadata, dict):
                    report.update({
                        "status": "source_metadata_retrieved_not_governance_labels",
                        "source": url,
                        "name": metadata.get("name"),
                        "doi": metadata.get("doi"),
                        "license": metadata.get("license"),
                        "num_instances": metadata.get("num_instances"),
                        "num_features": metadata.get("num_features"),
                    })
                    break
            else:
                report.update({"status": "official_dataset_page_reachable_metadata_unparsed", "source": url})
                break
        except (HTTPError, URLError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
            attempts.append({"url": url, "status": "failed", "error": str(exc)[:240]})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if report["status"] == "source_access_unverified":
        print("WARNING: UCI dataset endpoint unavailable; RCAEval audit remains independently valid.")

if __name__ == "__main__":
    main()
