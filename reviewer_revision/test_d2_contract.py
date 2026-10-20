"""Unit tests for D2 ranking, budget and fallback behavior."""
import copy
import unittest
from unittest.mock import patch

from reviewer_revision.d2_retrieval import rank_tools, run_d2


def missing(field: str) -> dict:
    domain, name = field.split(".", 1)
    return {domain: {name: None, "_evidence": {name: {"status": "missing"}}}}


class D2Tests(unittest.TestCase):
    def test_exact_mapping_first(self):
        evidence = missing("deploy.restart_burst_count")
        self.assertEqual(rank_tools(evidence, ["get_deployment_history", "get_restart_history"])[0], "get_restart_history")

    def test_unmapped_domain_fallback(self):
        evidence = missing("sre.unknown_signal")
        self.assertEqual(rank_tools(evidence, ["get_latency_history", "get_error_rate_history"])[0], "get_error_rate_history")

    def test_no_missing_fields_no_tools(self):
        self.assertEqual(rank_tools({"deploy": {"pipeline_failed": False}}, ["get_pipeline_history"]), [])

    def test_no_hidden_oracle_in_ranker(self):
        evidence = missing("deploy.restart_burst_count")
        tools = ["get_deployment_history", "get_restart_history"]
        baseline = rank_tools(evidence, tools)
        for oracle in ("get_restart_history", "get_deployment_history"):
            case = {"evidence": copy.deepcopy(evidence), "available_tools": tools, "expected_tool": oracle}
            self.assertEqual(rank_tools(case["evidence"], case["available_tools"]), baseline)

    def test_fallback_after_no_result_and_budget(self):
        case = {
            "evidence": missing("deploy.restart_burst_count"),
            "available_tools": ["get_restart_history", "get_deployment_history"],
            "expected_tool": "get_deployment_history",
            "tool_result": {"id": "E-1", "deploy": {"restart_burst_count": 7}},
        }
        with patch("reviewer_revision.d2_retrieval.deterministic_assessment",
                   return_value={"selected_action": "Mitigate and monitor"}):
            result = run_d2(case, max_calls=2)
        self.assertEqual(result["requested_tools"], ["get_restart_history", "get_deployment_history"])
        self.assertEqual(result["llm_calls"], 0)
        self.assertEqual(case["evidence"], missing("deploy.restart_burst_count"))

    def test_zero_budget(self):
        case = {"evidence": missing("deploy.restart_burst_count"), "available_tools": ["get_restart_history"]}
        with patch("reviewer_revision.d2_retrieval.deterministic_assessment",
                   return_value={"selected_action": "Escalate for evidence/human review"}):
            result = run_d2(case, max_calls=0)
        self.assertEqual(result["requested_tools"], [])

    def test_negative_budget(self):
        with self.assertRaises(ValueError):
            run_d2({"evidence": {}, "available_tools": []}, max_calls=-1)


if __name__ == "__main__":
    unittest.main()
