"""Stage 2 held-out tool execution with independent, per-tool responses.

Unlike the historical simulator, execution never consults expected_tool,
admissible_actions, case IDs or other evaluator-only oracle labels.
The caller must provide a separately curated response table. Every listed
tool has its own explicit response, including no-result and failure cases.
"""
from __future__ import annotations

import copy
from typing import Any


class ToolExecutionError(ValueError):
    """Invalid tool invocation or malformed independent response table."""


def execute_tool(tool: str, available_tools: list[str], tool_responses: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if tool not in available_tools:
        raise ToolExecutionError("Requested tool is not in the visible catalog")
    if tool not in tool_responses:
        raise ToolExecutionError("Missing independent response for available tool: " + tool)
    response = tool_responses[tool]
    if not isinstance(response, dict):
        raise ToolExecutionError("Tool response must be an object")
    return copy.deepcopy(response)


def validate_tool_fixture(available_tools: list[str], tool_responses: dict[str, dict[str, Any]]) -> None:
    if len(available_tools) != len(set(available_tools)):
        raise ToolExecutionError("Duplicate available tool")
    if set(available_tools) != set(tool_responses):
        raise ToolExecutionError("Every available tool must have exactly one response")
    for tool in available_tools:
        if not isinstance(tool_responses[tool], dict):
            raise ToolExecutionError("Tool response must be an object")


def visible_case_payload(case: dict[str, Any]) -> dict[str, Any]:
    """Minimal case view for policies; deliberately exclude all evaluator metadata."""
    return {
        "evidence": copy.deepcopy(case["evidence"]),
        "available_tools": copy.deepcopy(case["available_tools"]),
    }
