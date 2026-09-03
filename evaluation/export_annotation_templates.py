from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import yaml


DOMAIN_OPTIONS = "DevOps|SRE|FinOps|DevSecOps"
ACTION_OPTIONS = (
    "Rollback to stable deployment|Block release and fix pipeline|"
    "Mitigate and monitor|Scale adjustment|Patch or block release|"
    "No action (observe)|Other"
)


def _write(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "case_id", "source", "case_text", "evidence_json",
        "annotator_primary_domain", "annotator_secondary_domains",
        "annotator_action", "annotator_confidence_1_to_5", "annotator_notes",
        "allowed_domains", "allowed_actions",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def export_pm_prompts(repo_root: Path) -> List[Dict[str, Any]]:
    data = yaml.safe_load((repo_root / "prompts" / "pm_prompt_library.yaml").read_text(encoding="utf-8"))
    prompts = data["prompts"] if isinstance(data, dict) else data
    rows = []
    for i, item in enumerate(prompts, 1):
        rows.append({
            "case_id": item.get("id", f"PM-{i:03d}"),
            "source": "pm_prompt",
            "case_text": item.get("prompt", ""),
            "evidence_json": "",
            "annotator_primary_domain": "",
            "annotator_secondary_domains": "",
            "annotator_action": "",
            "annotator_confidence_1_to_5": "",
            "annotator_notes": "",
            "allowed_domains": DOMAIN_OPTIONS,
            "allowed_actions": ACTION_OPTIONS,
        })
    return rows


def export_controlled_scenarios(repo_root: Path) -> List[Dict[str, Any]]:
    # Generate scenarios but deliberately exclude designer-authored ground truth.
    from scenario_generator.generate import generate_scenarios
    scenarios = generate_scenarios(seed=42, noise={})
    rows = []
    for i, sc in enumerate(scenarios, 1):
        rows.append({
            "case_id": sc.get("incident_id", f"SC-{i:03d}"),
            "source": "controlled_scenario",
            "case_text": sc.get("scenario_type", "Controlled operational evidence case"),
            "evidence_json": json.dumps(sc.get("telemetry", {}), ensure_ascii=False, sort_keys=True),
            "annotator_primary_domain": "",
            "annotator_secondary_domains": "",
            "annotator_action": "",
            "annotator_confidence_1_to_5": "",
            "annotator_notes": "",
            "allowed_domains": DOMAIN_OPTIONS,
            "allowed_actions": ACTION_OPTIONS,
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Export blinded annotation sheets")
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--out", default="annotation_templates")
    args = ap.parse_args()
    root = Path(args.repo_root).resolve()
    out = root / args.out
    _write(out / "controlled_scenarios_blinded.csv", export_controlled_scenarios(root))
    _write(out / "pm_prompts_blinded.csv", export_pm_prompts(root))
    print(f"Blinded annotation templates written to {out}")


if __name__ == "__main__":
    main()
