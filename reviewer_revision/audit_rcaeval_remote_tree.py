"""Inspect RCAEval RE2 remote telemetry structure without downloading raw logs.

Fetch the public Hugging Face dataset repository tree (metadata only). Report
candidate paths and file sizes for each RE2 system. This is not a case-quality
or governance-label assessment.
"""
from __future__ import annotations
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen

BASE = "https://huggingface.co/api/datasets/phamquiluan/RCAEval/tree/main"
OUT = Path(__file__).resolve().parent / "results/external_dataset_audit/rcaeval_remote_tree.json"

def get_json(url: str):
    req = Request(url, headers={"User-Agent": "ieee-access-dataset-audit/1.0"})
    with urlopen(req, timeout=60) as response:
        return json.load(response)

def main():
    # HF tree endpoint is paginated and may return a continuation cursor.
    cursor = None
    entries = []
    for _ in range(40):
        url = BASE + "?recursive=true&expand=false&limit=1000"
        if cursor:
            url += "&cursor=" + quote(cursor, safe="")
        req = Request(url, headers={"User-Agent": "ieee-access-dataset-audit/1.0"})
        with urlopen(req, timeout=90) as response:
            page = json.load(response)
            link = response.headers.get("Link", "")
        if not isinstance(page, list):
            raise ValueError("Unexpected Hugging Face tree response")
        entries.extend(page)
        # Avoid guessing continuation from listing size: use RFC 5988 Link.
        import re
        match = re.search(r'[?&]cursor=([^&>]+)[^>]*>;\\s*rel="next"', link)
        if not match:
            break
        from urllib.parse import unquote
        cursor = unquote(match.group(1))
    else:
        raise RuntimeError("Pagination exceeded 40 pages; audit incomplete")
    files = [x for x in entries if x.get("type") == "file"]
    re2 = [x for x in files if "RE2" in str(x.get("path", "")).upper()]
    systems = {}
    for system in ("OB", "SS", "TT"):
        matches = [x for x in re2 if ("RE2-" + system) in str(x.get("path", "")).upper()]
        systems[system] = {
            "indexed_paths": len(matches),
            "sample_paths": [x.get("path") for x in matches[:12]],
            "total_indexed_bytes": sum(int(x.get("size") or 0) for x in matches),
        }
    report = {
        "status": "remote_tree_metadata_only_not_telemetry_inspection",
        "retrieved_utc": datetime.now(timezone.utc).isoformat(),
        "source": BASE,
        "tree_entries_seen": len(entries),
        "files_seen": len(files),
        "re2_files_seen": len(re2),
        "top_level_paths": dict(Counter(str(x.get("path", "")).split("/")[0] for x in files)),
        "re2_systems": systems,
        "re2_sample_paths": [x.get("path") for x in re2[:25]],
        "limitation": "Tree metadata cannot establish telemetry contents, completeness, license, or action labels.",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
