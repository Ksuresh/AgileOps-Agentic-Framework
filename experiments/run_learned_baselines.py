from __future__ import annotations

"""Independent learned baselines for the frozen HRT-01..HRT-13 evaluation.

Protocol safeguards
-------------------
* Training uses controlled/calibration scenarios only.
* HRT-01..HRT-13 are never used for fitting, feature selection, thresholding,
  hyperparameter search, or model selection.
* Hyperparameters below are fixed a priori and intentionally simple.
* The held-out CSV was frozen before this script was added.
* Results are reported at the scenario-template level (n=13).
"""

import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from scenario_generator.generate import generate_scenarios

OUT = Path("results_learned_baselines")
TEST_CSV = Path("experiments/learned_baseline_hrt_template_features.csv")

RAW_FEATURES: List[str] = [
    "deploy.restart_loops",
    "deploy.restart_burst_count",
    "deploy.restart_window_seconds",
    "deploy.pipeline_failed",
    "deploy.config_drift",
    "deploy.artifact_mismatch",
    "deploy.rollback_marker",
    "sre.p95_latency_ms",
    "sre.error_rate_pct",
    "sre.saturation_pct",
    "sre.availability_pct",
    "finops.cost_spike_pct",
    "finops.hpa_scale_to",
    "finops.cpu_request_increase_pct",
    "finops.memory_request_increase_pct",
    "sec.critical_cves",
    "sec.policy_violation",
    "sec.iam_drift",
    "sec.compliance_gap",
]

# Frozen Full-AAF template outcomes from the confirmatory evaluation. These are
# used only for paired reporting after the learned baselines have predicted.
AAF_FULL_MATCH = {
    "HRT-01": True, "HRT-02": True, "HRT-03": True, "HRT-04": True,
    "HRT-05": False, "HRT-06": True, "HRT-07": True, "HRT-08": True,
    "HRT-09": True, "HRT-10": True, "HRT-11": True, "HRT-12": True,
    "HRT-13": True,
}


def canonical_action(action: str) -> str:
    a = str(action).strip()
    aliases = {
        "No action": "No action (observe)",
        "Observe": "No action (observe)",
        "Rollback": "Rollback to stable deployment",
        "Block release/fix pipeline": "Block release and fix pipeline",
        "Block release and fix pipeline": "Block release and fix pipeline",
        "Patch/block release": "Patch or block release",
    }
    return aliases.get(a, a)


def _nested_get(d: Dict[str, Any], key: str) -> Any:
    domain, field = key.split(".", 1)
    block = d.get(domain, {}) or {}
    if block.get("_missing"):
        return None
    return block.get(field)


def flatten_training(telemetry: Dict[str, Any]) -> Dict[str, float | None]:
    out: Dict[str, float | None] = {}
    for key in RAW_FEATURES:
        value = _nested_get(telemetry, key)
        if value is None:
            out[key] = None
        elif isinstance(value, bool):
            out[key] = float(value)
        elif isinstance(value, (int, float)):
            out[key] = float(value)
        else:
            out[key] = None
    return out


def controlled_training_rows() -> List[Tuple[Dict[str, float | None], str, str]]:
    """Create 120 calibration instances as four noisy realizations of 30
    controlled templates. These are training instances, not independent test
    observations. They are never mixed with the HRT held-out templates.
    """
    rows: List[Tuple[Dict[str, float | None], str, str]] = []
    for seed in (42, 43, 44, 45):
        scenarios = generate_scenarios(
            seed=seed,
            noise={
                "missing_evidence_prob": 0.20,
                "contradiction_prob": 0.10,
                "metric_jitter_pct": 0.05,
            },
        )
        for sc in scenarios:
            y = canonical_action(sc["ground_truth"]["expected_action"])
            rows.append((flatten_training(sc["telemetry"]), y, sc["scenario_id"]))
    return rows


def load_heldout() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with TEST_CSV.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            feat: Dict[str, float | None] = {}
            for key in RAW_FEATURES:
                raw = row.get(key, "")
                feat[key] = None if raw in (None, "") else float(raw)
            rows.append({
                "case_id": row["case_id"],
                "oracle_actions": [canonical_action(x) for x in row["oracle_actions"].split("|")],
                "features": feat,
            })
    return rows


def matrix(rows: Sequence[Dict[str, float | None]], features: Sequence[str]) -> np.ndarray:
    return np.asarray([[np.nan if r.get(k) is None else float(r[k]) for k in features] for r in rows], dtype=float)


def exact_two_sided_binomial(n10: int, n01: int) -> float:
    d = n10 + n01
    if d == 0:
        return 1.0
    m = min(n10, n01)
    tail = sum(math.comb(d, k) for k in range(m + 1)) / (2 ** d)
    return min(1.0, 2.0 * tail)


def paired_vs_aaf(case_rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    aaf_only = sum(AAF_FULL_MATCH[r["case_id"]] and not r["match"] for r in case_rows)
    baseline_only = sum((not AAF_FULL_MATCH[r["case_id"]]) and r["match"] for r in case_rows)
    both_correct = sum(AAF_FULL_MATCH[r["case_id"]] and r["match"] for r in case_rows)
    both_wrong = sum((not AAF_FULL_MATCH[r["case_id"]]) and (not r["match"]) for r in case_rows)
    return {
        "both_correct": both_correct,
        "aaf_only_correct": aaf_only,
        "baseline_only_correct": baseline_only,
        "both_incorrect": both_wrong,
        "exact_two_sided_p": exact_two_sided_binomial(aaf_only, baseline_only),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    train = controlled_training_rows()
    heldout = load_heldout()

    # A raw feature that is absent for every training instance cannot be learned
    # from without leaking held-out information. Drop such columns based only on
    # the training set and record that decision explicitly.
    train_features = [r[0] for r in train]
    active_features = [
        k for k in RAW_FEATURES
        if any(row.get(k) is not None for row in train_features)
    ]
    dropped_features = [k for k in RAW_FEATURES if k not in active_features]

    X_train = matrix(train_features, active_features)
    y_train = np.asarray([r[1] for r in train], dtype=object)
    X_test = matrix([r["features"] for r in heldout], active_features)

    models = {
        "logistic_regression": Pipeline([
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(
                C=1.0,
                max_iter=5000,
                class_weight="balanced",
                random_state=42,
            )),
        ]),
        "decision_tree": Pipeline([
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("model", DecisionTreeClassifier(
                max_depth=4,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=42,
            )),
        ]),
    }

    summary: Dict[str, Any] = {
        "protocol": "trained on controlled/calibration scenarios only; frozen HRT templates used once for evaluation; no HRT tuning",
        "training_instances": len(train),
        "training_template_families": 30,
        "training_seeds": [42, 43, 44, 45],
        "heldout_templates": len(heldout),
        "active_features": active_features,
        "dropped_training_absent_features": dropped_features,
        "label_distribution": {str(k): int(v) for k, v in zip(*np.unique(y_train, return_counts=True))},
        "models": {},
    }

    all_case_rows: List[Dict[str, Any]] = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        case_rows: List[Dict[str, Any]] = []
        for r, p in zip(heldout, pred):
            p = canonical_action(str(p))
            match = p in set(r["oracle_actions"])
            case_rows.append({
                "model": name,
                "case_id": r["case_id"],
                "prediction": p,
                "oracle_actions": json.dumps(r["oracle_actions"]),
                "match": bool(match),
                "aaf_full_match": bool(AAF_FULL_MATCH[r["case_id"]]),
            })
        all_case_rows.extend(case_rows)
        summary["models"][name] = {
            "agreement_count": sum(r["match"] for r in case_rows),
            "agreement_rate": sum(r["match"] for r in case_rows) / len(case_rows),
            "paired_vs_full_aaf": paired_vs_aaf(case_rows),
        }

    with (OUT / "learned_baseline_case_results.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(all_case_rows[0].keys()))
        w.writeheader(); w.writerows(all_case_rows)

    (OUT / "learned_baseline_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
