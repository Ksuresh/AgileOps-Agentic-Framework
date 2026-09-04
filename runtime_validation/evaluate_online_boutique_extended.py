from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results_online_boutique_runtime"


def main() -> None:
    rows = []
    for i in range(9, 17):
        p = OUT / f"OB-{i:02d}.json"
        if not p.exists():
            raise FileNotFoundError(p)
        rows.append(json.loads(p.read_text(encoding="utf-8")))

    n = len(rows)
    summary = {
        "runtime_system": "Google Online Boutique",
        "source_commit": "b9a978db9e01f4ad3dca9494a22cb9edc17548fe",
        "evaluation_reference": "frozen OB-09--OB-16 extension",
        "n_independent_scenario_templates": n,
        "dominant_domain_baseline_action_oracle_agreement": sum(r["dominant_domain_baseline_match"] for r in rows) / n,
        "aaf_no_interaction_action_oracle_agreement": sum(r["aaf_no_interaction_match"] for r in rows) / n,
        "aaf_full_action_oracle_agreement": sum(r["aaf_full_match"] for r in rows) / n,
        "aaf_only_wins_vs_dominant_domain": sum(r["aaf_full_match"] and not r["dominant_domain_baseline_match"] for r in rows),
        "dominant_domain_only_wins": sum(r["dominant_domain_baseline_match"] and not r["aaf_full_match"] for r in rows),
        "both_correct_vs_dominant_domain": sum(r["aaf_full_match"] and r["dominant_domain_baseline_match"] for r in rows),
        "both_incorrect_vs_dominant_domain": sum((not r["aaf_full_match"]) and (not r["dominant_domain_baseline_match"]) for r in rows),
        "interpretation": "Eight newly frozen independent Online Boutique templates using the same frozen AAF decision policy and NodePort evidence path; no post-outcome retuning.",
    }
    (OUT / "extended_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
