# Study 10b: third-model replication (frozen before any run)

Same design, cases, arms, runs and analysis as Study 10 (`PROTOCOL.md`), with a third model family:
- Model: `llama3.1:8b` (Meta Llama 3.1 8B Instruct, Q4_K_M) via Ollama 0.40.9 on GitHub CPU runners; digest recorded per shard.
- Llama 3.1 has no thinking mode, so B6 runs with its policy-aware prompt but without thinking (`NO_THINK=1`); this is
  reported as a difference from Study 10.
- Hypotheses: H10.1 to H10.4 as in Study 10, with GPT vs Llama as the model pair; additionally the proportion of the 96 cases
  on which each arm's majority action is identical under all three models (GPT, Qwen, Llama).
