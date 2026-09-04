from pathlib import Path

from runtime_validation import run_batch3_case as batch3

# Reuse the audited Batch-3 runtime harness. Only the frozen manifest changes.
batch3.MANIFEST = Path(__file__).with_name("interventions_extended_validation.yaml")

if __name__ == "__main__":
    batch3.main()
