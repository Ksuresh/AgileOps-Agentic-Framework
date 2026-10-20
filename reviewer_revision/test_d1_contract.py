"""Contract tests for deterministic retrieval; no LLM calls or oracle dependence."""
import copy
import unittest
from unittest.mock import patch

from reviewer_revision.deterministic_retrieval import (
    run_deterministic_retrieval, select_tools, simulated_tool_execution,
)

MISSING = {
    "deploy": {
        "restart_burst_count": None,
        "restart_window_seconds": None,
        "_evidence": {
            "restart_burst_count": {"status": "missing"},
            "restart_window_seconds": {"status": "missing"},
        },
    }
}

class RetrievalContractTests(unittest.TestCase):
    def test_selector_uses_only_visible_evidence_and_catalog(self):
        available = ["get_restart_history", "get_policy_gate_status"]
        selected = select_tools(copy.deepcopy(MISSING), available)
        self.assertEqual(selected, ["get_restart_history"])
        self.assertEqual(select_tools(MISSING, ["get_policy_gate_status"]), [])
        self.assertEqual(select_tools(MISSING, available, max_calls=0), [])

    def test_selector_does_not_depend_on_hidden_oracle(self):
        visible = copy.deepcopy(MISSING)
        catalog = ["get_restart_history", "get_policy_gate_status"]
        baseline = select_tools(visible, catalog)
        for hidden in (
            {"expected_tool": "get_restart_history", "admissible_actions": ["A"], "id": "one"},
            {"expected_tool": "get_policy_gate_status", "admissible_actions": ["B"], "id": "two"},
        ):
            # Selector API cannot accept hidden evaluator fields.
            case = {"evidence": visible, "available_tools": catalog, **hidden}
            self.assertEqual(select_tools(case["evidence"], case["available_tools"]), baseline)

    def test_no_result_for_wrong_tool(self):
        case = {
            "available_tools": ["get_restart_history", "get_policy_gate_status"],
            "expected_tool": "get_policy_gate_status",
            "tool_result": {"id": "PRIVATE", "deploy": {"restart_burst_count": 8}},
        }
        response = simulated_tool_execution(case, "get_restart_history")
        self.assertEqual(response["status"], "no_additional_evidence")
        self.assertNotIn("PRIVATE", str(response))

    def test_reject_unavailable_tool(self):
        with self.assertRaises(ValueError):
            simulated_tool_execution({"available_tools": []}, "get_restart_history")

    def test_budget_and_no_mutation(self):
        original = copy.deepcopy(MISSING)
        self.assertEqual(select_tools(MISSING, ["get_restart_history"], max_calls=1), ["get_restart_history"])
        self.assertEqual(MISSING, original)
        with self.assertRaises(ValueError):
            select_tools(MISSING, [], max_calls=-1)

    def test_no_result_does_not_merge_oracle_response(self):
        case = {
            "evidence": copy.deepcopy(MISSING),
            "available_tools": ["get_restart_history"],
            "expected_tool": "get_policy_gate_status",
            "tool_result": {"id": "HIDDEN", "deploy": {"restart_burst_count": 9}},
        }
        with patch("reviewer_revision.deterministic_retrieval.deterministic_assessment",
                   return_value={"selected_action": "Escalate for evidence/human review"}):
            result = run_deterministic_retrieval(case)
        self.assertEqual(result["requested_tools"], ["get_restart_history"])
        self.assertEqual(result["missing_fields_after"], result["missing_fields_before"])
        self.assertEqual(result["llm_calls"], 0)
        self.assertEqual(case["evidence"], MISSING)

if __name__ == "__main__":
    unittest.main()
