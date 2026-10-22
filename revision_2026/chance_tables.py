"""R5-2: uniform-chance and best-constant reference levels for the exact case sets of manuscript Tables 4-7.

Uniform chance = expected agreement of a uniformly random choice over the 8-action vocabulary.
Best constant = agreement of always choosing the single most frequently admissible action.
OB-09..OB-16 admissible sets are read from the frozen extension commit fdf222fc (copied to inputs/)."""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agentic_experiments.agentic_runtime import ACTIONS  # noqa: E402

V = len(ACTIONS)
RV = ROOT / "runtime_validation"


def load(path: Path) -> dict[str, list[str]]:
    d = yaml.safe_load(path.read_text())
    items = d.get("cases") or d.get("interventions") or d.get("templates") if isinstance(d, dict) else d
    if isinstance(items, dict):
        items = list(items.values())
    out = {}
    for it in items:
        if isinstance(it, dict):
            s = it.get("admissible_actions") or (it.get("oracle") or {}).get("admissible_actions")
            i = it.get("template_id") or it.get("id") or it.get("case_id") or it.get("name")
            if s and i not in out:
                out[i] = list(s)
    return out


def ref(name: str, sets: list[list[str]]) -> dict:
    freq = Counter(a for s in sets for a in s)
    k, a = max((sum(x in s for s in sets), x) for x in freq)
    return {"table_set": name, "n": len(sets), "uniform_chance_pct": round(100 * sum(map(len, sets)) / (len(sets) * V), 1),
            "best_constant_action": a, "best_constant_pct": round(100 * k / len(sets), 1)}


def main() -> None:
    rt = {}
    for f in sorted(RV.glob("interventions*.yaml")):
        for k, v in load(f).items():
            rt.setdefault(k, v)
    ob = {**load(RV / "interventions_online_boutique.yaml"),
          **load(Path(__file__).parent / "inputs/interventions_online_boutique_extended_fdf222fc.yaml")}
    res = [ref("Table 4: Sock Shop HRT-01..HRT-31", [rt[f"HRT-{i:02d}"] for i in range(1, 32)]),
           ref("Table 5: Online Boutique OB-01..OB-16", [ob[f"OB-{i:02d}"] for i in range(1, 17)]),
           ref("Table 6: prospective HRT-32..HRT-39", [rt[f"HRT-{i:02d}"] for i in range(32, 40)]),
           ref("Table 7: selective AEI primary (32)", list(load(ROOT / "agentic_experiments/cases_v1.yaml").values())),
           ref("Table 7: selective AEI replication (16)", list(load(ROOT / "agentic_experiments/cases_replication_v1.yaml").values()))]
    (Path(__file__).parent / "results/chance_tables.json").write_text(json.dumps(res, indent=1))
    for r in res:
        print(r)


if __name__ == "__main__":
    main()
