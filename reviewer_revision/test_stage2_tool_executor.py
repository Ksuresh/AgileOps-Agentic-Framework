"""Stage 2 tool executor isolation tests."""
import unittest

from reviewer_revision.stage2_tool_executor import (
    ToolExecutionError, execute_tool, validate_tool_fixture, visible_case_payload,
)


class Stage2ToolExecutorTests(unittest.TestCase):
    def setUp(self):
        self.catalog = ["get_restart_history", "get_pipeline_history"]
        self.responses = {
            "get_restart_history": {"id": "E1", "deploy": {"restart_burst_count": 6}},
            "get_pipeline_history": {"status": "no_additional_evidence"},
        }

    def test_independent_responses(self):
        self.assertEqual(execute_tool("get_restart_history", self.catalog, self.responses)["id"], "E1")
        self.assertEqual(execute_tool("get_pipeline_history", self.catalog, self.responses)["status"], "no_additional_evidence")

    def test_copy_prevents_mutation(self):
        result = execute_tool("get_restart_history", self.catalog, self.responses)
        result["deploy"]["restart_burst_count"] = 999
        self.assertEqual(self.responses["get_restart_history"]["deploy"]["restart_burst_count"], 6)

    def test_no_unlisted_tools(self):
        with self.assertRaises(ToolExecutionError):
            execute_tool("get_secret", self.catalog, self.responses)

    def test_incomplete_response_table_rejected(self):
        with self.assertRaises(ToolExecutionError):
            validate_tool_fixture(self.catalog, {"get_restart_history": {}})

    def test_valid_response_table(self):
        validate_tool_fixture(self.catalog, self.responses)

    def test_hidden_oracles_not_in_visible_payload(self):
        case = {
            "id": "PRIVATE", "stratum": "INCOMPLETE",
            "evidence": {"deploy": {"pipeline_failed": False}},
            "available_tools": self.catalog,
            "expected_tool": "get_restart_history",
            "tool_result": {"secret": 1},
            "admissible_actions": ["Rollback to stable deployment"],
            "oracle_domains": ["DevOps"],
        }
        visible = visible_case_payload(case)
        self.assertEqual(set(visible), {"evidence", "available_tools"})
        self.assertNotIn("PRIVATE", str(visible))
        self.assertNotIn("secret", str(visible))


if __name__ == "__main__":
    unittest.main()
