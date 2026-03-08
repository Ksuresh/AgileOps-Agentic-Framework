from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import yaml

from aaf.utils import set_seed, now_ms
from scenario_generator.generate import generate_scenarios
from pipeline import run_once
from llm.llama_cpp_runner import run_llama


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "config" / "config.yaml"


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run AgileOps Agentic Framework reproducibility experiments")
    ap.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="Path to config YAML file",
    )
    ap.add_argument(
        "--llm",
        action="store_true",
        help="Enable local llama.cpp-based explanation generation",
    )
    ap.add_argument(
        "--no-llm",
        action="store_true",
        help="Force-disable LLM explanation generation",
    )
    ap.add_argument(
        "--llama_bin",
        default="",
        help="Path to llama.cpp executable",
    )
    ap.add_argument(
        "--gguf",
        default="",
        help="Path to GGUF model file",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    use_llm = bool(args.llm and not args.no_llm)

    seed = int(cfg["experiment"]["random_seed"])
    set_seed(seed)

    noise = cfg["experiment"]["scenario_noise"]
    scenarios = generate_scenarios(seed=seed, noise=noise)

    lam = float(cfg["experiment"].get("lambda", 0.5))

    utility_cfg = cfg["experiment"]["utility_weights"]
    w = (
        float(utility_cfg["w_perf"]),
        float(utility_cfg["w_cost"]),
        float(utility_cfg["w_risk"]),
    )

    if abs(sum(w) - 1.0) > 1e-6:
        raise ValueError("Utility weights must sum to 1.0")

    outdir = ROOT / "results" / f"run_{int(time.time())}"
    outdir.mkdir(parents=True, exist_ok=True)

    sys_prompt = load_text(ROOT / "prompts" / "system.txt")
    user_template = load_text(ROOT / "prompts" / "user_template.txt")

    lat_rows = []
    outputs_path = outdir / "scenario_outputs.jsonl"

    with outputs_path.open("w", encoding="utf-8") as f:
        for sc in scenarios:
            t0 = now_ms()

            # Telemetry parse stage
            t_parse0 = now_ms()
            telemetry = sc["telemetry"]
            t_parse = now_ms() - t_parse0

            # Pipeline stage
            t_pipe0 = now_ms()
            res = run_once(
                telemetry=telemetry,
                thresholds=cfg["experiment"]["thresholds"],
                lam=lam,
                w=w,
            )
            t_pipe = now_ms() - t_pipe0

            # Optional LLM explanation
            llm_text = ""
            t_llm = 0

            if use_llm:
                if not args.llama_bin or not args.gguf:
                    raise ValueError("Provide --llama_bin and --gguf when using --llm")

                user_prompt = user_template.format(
                    incident_id=sc["incident_id"],
                    serialized_agent_json=json.dumps(res, indent=2),
                    consensus_score=res["consensus_score"],
                    rar_triggered=res["rar_triggered"],
                    rar_loops=res["rar_loops"],
                    recommended_action=res["recommended_action"],
                    utility_score=res["utility_score"],
                )

                t_llm0 = now_ms()
                llm_text = run_llama(
                    llama_bin=args.llama_bin,
                    gguf_path=args.gguf,
                    system_prompt=sys_prompt,
                    user_prompt=user_prompt,
                    temperature=float(cfg["llm"]["temperature"]),
                    top_p=float(cfg["llm"]["top_p"]),
                    max_tokens=int(cfg["llm"]["max_tokens"]),
                )
                t_llm = now_ms() - t_llm0

            total = now_ms() - t0

            lat_rows.append(
                {
                    "incident_id": sc["incident_id"],
                    "scenario_type": sc["scenario_type"],
                    "telemetry_parse_ms": t_parse,
                    "pipeline_ms": t_pipe,
                    "llm_ms": t_llm,
                    "total_ms": total,
                    "rar_triggered": res["rar_triggered"],
                    "rar_loops": res["rar_loops"],
                    "consensus_score": res["consensus_score"],
                    "utility_score": res["utility_score"],
                }
            )

            payload = {
                "incident_id": sc["incident_id"],
                "scenario_type": sc["scenario_type"],
                "ground_truth": sc["ground_truth"],
                **res,
                "llm_output": llm_text.strip()[:4000] if llm_text else "",
            }

            f.write(json.dumps(payload) + "\n")

    df = pd.DataFrame(lat_rows)
    df.to_csv(outdir / "latency.csv", index=False)

    summary = {
        "n": int(len(df)),
        "random_seed": seed,
        "lambda": lam,
        "llm_enabled": use_llm,
        "embedding_method": cfg.get("embeddings", {}).get("method", "unknown"),
        "embedding_model": cfg.get("embeddings", {}).get("model", "unknown"),
        "rar_trigger_rate": float(df["rar_triggered"].mean()),
        "consensus_mean": float(df["consensus_score"].mean()),
        "utility_mean": float(df["utility_score"].mean()),
        "latency_total_ms_mean": float(df["total_ms"].mean()),
        "latency_total_ms_p50": float(df["total_ms"].quantile(0.50)),
        "latency_total_ms_p95": float(df["total_ms"].quantile(0.95)),
        "latency_total_ms_p99": float(df["total_ms"].quantile(0.99)),
    }

    (outdir / "metrics_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print(f"Results written to: {outdir}")


if __name__ == "__main__":
    main()
