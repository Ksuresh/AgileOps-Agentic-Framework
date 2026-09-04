from pathlib import Path

from runtime_validation import run_batch3_case as batch3

# Reuse the audited Sock Shop intervention runner; only the frozen prospective
# manifest is changed. AAF-v1 and AAF-v2 decision logic are not read by the
# intervention runner.
batch3.MANIFEST = Path(__file__).with_name("interventions_v2_prospective.yaml")

if __name__ == "__main__":
    batch3.main()
