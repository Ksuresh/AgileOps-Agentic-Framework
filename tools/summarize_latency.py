from __future__ import annotations
import sys
import pandas as pd

def main():
    if len(sys.argv) < 2:
        print("Usage: python tools/summarize_latency.py results/.../latency.csv")
        raise SystemExit(2)
    df = pd.read_csv(sys.argv[1])
    for col in ["telemetry_parse_ms","pipeline_ms","llm_ms","total_ms"]:
        if col not in df.columns:
            continue
        print("\n" + col)
        print("mean:", df[col].mean())
        print("std :", df[col].std())
        print("p50 :", df[col].quantile(0.50))
        print("p95 :", df[col].quantile(0.95))
        print("p99 :", df[col].quantile(0.99))

if __name__ == "__main__":
    main()
