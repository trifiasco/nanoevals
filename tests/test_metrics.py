import pytest
from nanoevals.types import Trace, ToolCall
from nanoevals.dataset import AgentTestCase, ExpectedToolCall
from nanoevals.metrics import tool_correctness, trajectory_match


def _make_trace(*tool_names: str) -> Trace:
    return Trace(
        input="test",
        output="test",
        tool_calls=[ToolCall(name=n) for n in tool_names],
    )


def _make_case(*tool_names: str) -> AgentTestCase:
    return AgentTestCase(
        input="test",
        expected_trajectory=[ExpectedToolCall(name=n) for n in tool_names],
    )


def test_tool_correctness_all_match():
    trace = _make_trace("a", "b")
    case = _make_case("a", "b")
    result = tool_correctness(trace, case)
    assert result.score == 1.0
    assert result.passed


def test_tool_correctness_partial():
    trace = _make_trace("a", "c")
    case = _make_case("a", "b")
    result = tool_correctness(trace, case)
    assert result.score == 0.5
    assert not result.passed


def test_tool_correctness_fuzzy():
    trace = Trace(
        input="test",
        output="test",
        tool_calls=[ToolCall(name="search", args={"q": "hello", "limit": 10})],
    )
    case = AgentTestCase(
        input="test",
        expected_trajectory=[
            ExpectedToolCall(name="search", args={"q": "hello"}, match_mode="fuzzy")
        ],
    )
    result = tool_correctness(trace, case)
    assert result.score == 1.0


def test_trajectory_match_exact_order():
    trace = _make_trace("a", "b", "c")
    case = _make_case("a", "b", "c")
    result = trajectory_match(trace, case)
    assert result.score == 1.0


def test_trajectory_match_wrong_order():
    trace = _make_trace("c", "b", "a")
    case = _make_case("a", "b", "c")
    result = trajectory_match(trace, case)
    assert result.score == pytest.approx(1 / 3)


def test_trajectory_match_partial():
    trace = _make_trace("a", "x", "b", "y", "c")
    case = _make_case("a", "b", "c")
    result = trajectory_match(trace, case)
    assert result.score == 1.0
