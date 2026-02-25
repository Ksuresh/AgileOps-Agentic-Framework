from __future__ import annotations
import argparse, json, time
from pathlib import Path
import yaml
import pandas as pd

from aaf.utils import set_seed, now_ms
from scenario_generator.generate import generate_scenarios
from pipeline import run_once
from llm.llama_cpp_runner import run_llama

def load_text(p: Path) -> str:
    return p.read_text(encoding="utf-8")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--llm", action="store_true")
    ap.add_argument("--llama_bin", default="")
    ap.add_argument("--gguf", default="")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    set_seed(int(cfg["experiment"]["random_seed"]))

    scenarios = generate_scenarios(cfg["experiment"]["random_seed"], cfg["experiment"]["scenario_noise"])

    outdir = Path("results") / f"run_{int(time.time())}"
    outdir.mkdir(parents=True, exist_ok=True)

    sys_prompt = load_text(Path("prompts/system.txt"))
    user_template = load_text(Path("prompts/user_template.txt"))

    lat_rows = []
    outputs_path = outdir / "scenario_outputs.jsonl"
    with outputs_path.open("w", encoding="utf-8") as f:
        for sc in scenarios:
            t0 = now_ms()

            # telemetry_parse (simulated)
            t_parse0 = now_ms()
            telemetry = sc["telemetry"]
            t_parse = now_ms() - t_parse0

            # pipeline
            t_pipe0 = now_ms()
            res = run_once(
                telemetry=telemetry,
                thresholds=cfg["experiment"]["thresholds"],
                lam=0.5,
                w=(cfg["experiment"]["utility_weights"]["w_perf"],
                   cfg["experiment"]["utility_weights"]["w_cost"],
                   cfg["experiment"]["utility_weights"]["w_risk"])
            )
            t_pipe = now_ms() - t_pipe0

            # LLM explanation
            llm_text = ""
            t_llm = 0.0
            if args.llm and not args.no_llm:
                assert args.llama_bin and args.gguf, "Provide --llama_bin and --gguf for LLM mode"
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
            lat_rows.append({
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
            })

            payload = {
                "incident_id": sc["incident_id"],
                "scenario_type": sc["scenario_type"],
                "ground_truth": sc["ground_truth"],
                **res,
                "llm_output": (llm_text.strip()[:4000] if llm_text else "")
            }
            f.write(json.dumps(payload) + "\n")

    df = pd.DataFrame(lat_rows)
    df.to_csv(outdir / "latency.csv", index=False)

    summary = {
        "n": int(len(df)),
        "rar_trigger_rate": float(df["rar_triggered"].mean()),
        "consensus_mean": float(df["consensus_score"].mean()),
        "utility_mean": float(df["utility_score"].mean()),
        "latency_total_ms_mean": float(df["total_ms"].mean()),
        "latency_total_ms_p50": float(df["total_ms"].quantile(0.50)),
        "latency_total_ms_p95": float(df["total_ms"].quantile(0.95)),
        "latency_total_ms_p99": float(df["total_ms"].quantile(0.99)),
    }
    (outdir / "metrics_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Wrote results to:", outdir)

if __name__ == "__main__":
    main()
