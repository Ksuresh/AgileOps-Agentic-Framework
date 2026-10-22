"""R5-2 / Theme D: admissible-set sizes, chance-level agreement and effective sample sizes."""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agentic_experiments.agentic_runtime import ACTIONS  # noqa: E402

OUT = Path(__file__).resolve().parent / "results"
V = len(ACTIONS)  # governance action vocabulary available to every arm (8 incl. escalate)


def block(name: str, sets: list[list[str]]) -> dict:
    sizes = [len(s) for s in sets]
    # chance = expected agreement of a uniform random choice over the vocabulary; majority-class = best constant guess
    freq = Counter(a for s in sets for a in s)
    best_const = max((sum(a in s for s in sets), a) for a in freq)
    return {"study": name, "n": len(sets), "admissible_set_sizes": dict(Counter(sizes)),
            "mean_admissible": round(sum(sizes) / len(sizes), 2), "vocabulary": V,
            "chance_uniform_pct": round(100 * sum(sizes) / (len(sizes) * V), 1),
            "best_constant_action": best_const[1], "best_constant_pct": round(100 * best_const[0] / len(sets), 1)}


def yaml_sets(path: Path, key="admissible_actions") -> list[list[str]]:
    d = yaml.safe_load(path.read_text())
    items = d.get("cases") or d.get("interventions") or d.get("templates") or d if isinstance(d, dict) else d
    if isinstance(items, dict):
        items = list(items.values())
    out = []
    for it in items:
        if isinstance(it, dict):
            s = it.get(key) or (it.get("oracle") or {}).get(key)
            if s:
                out.append(list(s))
    return out


def main() -> None:
    res = [block("selective AEI primary (32)", yaml_sets(ROOT / "agentic_experiments/cases_v1.yaml")),
           block("selective AEI replication (16)", yaml_sets(ROOT / "agentic_experiments/cases_replication_v1.yaml"))]
    for f in sorted((ROOT / "runtime_validation").glob("interventions*.yaml")):
        s = yaml_sets(f)
        if s:
            res.append(block(f"runtime {f.stem} ({len(s)})", s))
    (OUT / "chance_level.json").write_text(json.dumps(res, indent=1))
    for r in res:
        print(r)


if __name__ == "__main__":
    main()
