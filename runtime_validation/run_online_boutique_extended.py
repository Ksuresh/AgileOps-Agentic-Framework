from pathlib import Path

from runtime_validation import run_online_boutique_case_nodeport as nodeport

nodeport.impl.MANIFEST = Path(__file__).with_name("interventions_online_boutique_extended.yaml")

if __name__ == "__main__":
    nodeport.impl.main()
