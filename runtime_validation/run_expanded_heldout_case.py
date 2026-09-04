from pathlib import Path

from runtime_validation import run_batch3_case as batch3

# Reuse the already-audited Batch-3 intervention runner without changing the
# frozen AAF policy. Only the manifest is replaced with the newly frozen
# HRT-14--HRT-21 scenario set.
batch3.MANIFEST = Path(__file__).with_name("interventions_expanded_heldout.yaml")

if __name__ == "__main__":
    batch3.main()
