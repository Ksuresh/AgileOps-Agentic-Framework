"""Study 10: second-model replication with an open-weight model (Qwen via Ollama).

Same code paths, prompts, cases and arms as Studies 6-7; only the LLM backend changes. The runner calls Ollama's native
/api/chat with JSON output; "reasoning none" maps to think=false and "reasoning medium" (B6) to think=true.
Arms: AAF (hybrid.run_hybrid), B5 (run_agentic_only, original prompt), B6 (run_agentic_only, policy-aware prompt, thinking).
B3 is derived offline from B5 runs (deterministic re-decision), exactly as in Study 6.
Jobs are sharded (SHARD / NSHARDS) so a GitHub Actions matrix can run them in parallel on CPU runners.
"""
from __future__ import annotations

import json
import os
import random
import sys
import time
import urllib.request
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agentic_experiments import agentic_runtime as AR  # noqa: E402
from agentic_experiments.hybrid import run_agentic_only, run_hybrid  # noqa: E402
from revision_2026 import run_llm_arms as RA  # noqa: E402

MODEL = os.environ.get("QWEN_MODEL", "qwen3:8b")
HOST = os.environ.get("OLLAMA_HOST_URL", "http://127.0.0.1:11434")
TEMPERATURE = float(os.environ.get("QWEN_TEMPERATURE", "0.7"))
NUM_CTX = int(os.environ.get("QWEN_NUM_CTX", "8192"))
NUM_PREDICT = int(os.environ.get("QWEN_NUM_PREDICT", "3072"))
CASE_FILES = {
    "s6_primary": ROOT / "agentic_experiments" / "cases_v1.yaml",
    "s6_replication": ROOT / "agentic_experiments" / "cases_replication_v1.yaml",
    "s7_new": ROOT / "external_validation" / "expert_study" / "cases_new_v1.yaml",
}
RUNS = {"AAF": [1, 2, 3], "B5": [1, 2, 3], "B6": [1, 2]}


class QwenRuntime(AR.BoundedAgenticRuntime):
    def __init__(self, think: bool, coordinator_system: str | None = None):
        self.model = MODEL
        self.reasoning_effort = "medium" if think else "none"
        self.think = think and not os.environ.get("NO_THINK")  # models without a thinking mode (Study 10b)
        self.coordinator_system = coordinator_system or AR.COORDINATOR_SYSTEM
        self.errors: list[str] = []

    def _call(self, instructions, payload):
        if instructions == AR.COORDINATOR_SYSTEM:
            instructions = self.coordinator_system
        body = {
            "model": MODEL, "stream": False, "format": "json", "think": self.think,
            "options": {"temperature": TEMPERATURE, "num_ctx": NUM_CTX, "num_predict": NUM_PREDICT},
            "messages": [{"role": "system", "content": instructions},
                         {"role": "user", "content": json.dumps(payload, ensure_ascii=False, sort_keys=True)}],
        }
        req = urllib.request.Request(f"{HOST}/api/chat", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=1800) as r:
            body = r.read().decode("utf-8", "replace")
        try:
            out = json.loads(body)
        except json.JSONDecodeError:  # tolerate NDJSON (streamed) bodies: merge message chunks
            chunks = [json.loads(x) for x in body.splitlines() if x.strip().startswith("{")]
            if not chunks:
                self.errors.append(body[:300])
                raise
            out = dict(chunks[-1])
            out["message"] = {"content": "".join((c.get("message") or {}).get("content", "") for c in chunks)}
        latency = (time.perf_counter() - t0) * 1000.0
        raw = (out.get("message") or {}).get("content") or ""
        pin, pout = int(out.get("prompt_eval_count") or 0), int(out.get("eval_count") or 0)
        return AR._json_obj(raw), raw, {"input_tokens": pin, "output_tokens": pout, "total_tokens": pin + pout}, latency


def jobs() -> list[tuple[str, str, dict, int]]:
    out = []
    for study, path in CASE_FILES.items():
        for case in yaml.safe_load(path.read_text())["cases"]:
            for arm, runs in RUNS.items():
                for run in runs:
                    out.append((arm, study, case, run))
    random.Random(2026).shuffle(out)  # balance heavy (B6) and light (CLEAR) jobs across shards
    return out


def main() -> None:
    shard, nshards = int(os.environ.get("SHARD", "0")), int(os.environ.get("NSHARDS", "1"))
    limit = int(os.environ.get("LIMIT", "0"))  # smoke test: first LIMIT jobs of this shard
    tag = MODEL.replace(":", "_").replace("/", "_")
    out = ROOT / "model_replication" / "results" / tag / os.environ.get("RUN_LABEL", "run")
    out.mkdir(parents=True, exist_ok=True)
    rt = {"AAF": QwenRuntime(False), "B5": QwenRuntime(False), "B6": QwenRuntime(True, RA.COORD_B6)}
    fn = {"AAF": lambda c: run_hybrid(c, rt["AAF"]), "B5": lambda c: run_agentic_only(c, rt["B5"]),
          "B6": lambda c: run_agentic_only(c, rt["B6"])}
    arms = os.environ.get("ARMS", "").split()
    mine = [j for j in jobs() if not arms or j[0] in arms][shard::nshards]
    if limit:
        mine = mine[:limit]
    path = out / f"shard_{shard:02d}.jsonl"
    done = set()
    if path.exists():
        done = {(r["arm"], r["case"], r["run"]) for r in map(json.loads, path.read_text().splitlines()) if r.get("error") is None}
    with path.open("a") as fh:
        for arm, study, case, run in mine:
            if (arm, case["id"], run) in done:
                continue
            t0 = time.time()
            try:
                res = fn[arm](case)
                row = {"arm": arm, "study": study, "case": case["id"], "stratum": case["stratum"], "substratum": case.get("substratum"),
                       "run": run, "action": res["selected_action"], **RA.slim(res),
                       "trigger": (res.get("trigger") or {}).get("reasons"), "error": None}
            except Exception as exc:
                row = {"arm": arm, "study": study, "case": case["id"], "stratum": case["stratum"], "run": run, "action": None,
                       "error": f"{type(exc).__name__}: {str(exc)[:300]}"}
            row["backend_errors"] = rt[arm].errors[-3:]
            rt[arm].errors.clear()
            row.update({"model": MODEL, "temperature": TEMPERATURE, "wall_s": round(time.time() - t0, 1)})
            fh.write(json.dumps(row, default=str) + "\n"); fh.flush()
            print(arm, case["id"], run, row["action"], row["wall_s"], "s", flush=True)


if __name__ == "__main__":
    main()
