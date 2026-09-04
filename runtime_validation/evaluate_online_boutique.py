from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results_online_boutique_runtime"


def main() -> None:
    rows = []
    for path in sorted(OUT.glob("OB-*.json")):
        rows.append(json.loads(path.read_text(encoding="utf-8")))
    if len(rows) != 8:
        raise RuntimeError(f"expected 8 Online Boutique scenario results, found {len(rows)}")

    def rate(key: str) -> float:
        return sum(bool(r[key]) for r in rows) / len(rows)

    summary = {
        "runtime_system": "Google Online Boutique",
        "source_commit": "b9a978db9e01f4ad3dca9494a22cb9edc17548fe",
        "n_independent_scenario_templates": len(rows),
        "dominant_domain_baseline_action_oracle_agreement": rate("dominant_domain_baseline_match"),
        "aaf_no_interaction_action_oracle_agreement": rate("aaf_no_interaction_match"),
        "aaf_full_action_oracle_agreement": rate("aaf_full_match"),
        "aaf_only_wins_vs_dominant_domain": sum(r["aaf_full_match"] and not r["dominant_domain_baseline_match"] for r in rows),
        "dominant_domain_only_wins": sum(r["dominant_domain_baseline_match"] and not r["aaf_full_match"] for r in rows),
        "both_correct_vs_dominant_domain": sum(r["aaf_full_match"] and r["dominant_domain_baseline_match"] for r in rows),
        "both_incorrect_vs_dominant_domain": sum((not r["aaf_full_match"]) and (not r["dominant_domain_baseline_match"]) for r in rows),
        "interpretation": "Independent second-benchmark scenario-template analysis with frozen AAF policy and benchmark-specific evidence acquisition; not production accuracy.",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
