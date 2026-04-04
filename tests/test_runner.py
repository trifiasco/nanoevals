import tempfile
from nanoevals.types import Trace, ToolCall, UsageStats, EvalResult
from nanoevals.dataset import AgentTestCase, AgentGoldenDataset, ExpectedToolCall
from nanoevals.runner import run_eval, RunReport


def _mock_agent(input_text: str) -> Trace:
    return Trace(
        input=input_text,
        output=f"response to {input_text}",
        tool_calls=[ToolCall(name="search", args={"q": input_text})],
        usage=UsageStats(latency_ms=100.0, input_tokens=10, output_tokens=20, cost=0.001),
    )


def _mock_judge(trace: Trace, test_case: AgentTestCase) -> list[EvalResult]:
    return [EvalResult(metric="task_success", score=1.0, passed=True)]


DATASET = AgentGoldenDataset(
    test_cases=[
        AgentTestCase(
            input="hello",
            expected_trajectory=[ExpectedToolCall(name="search", args={"q": "hello"})],
        ),
        AgentTestCase(
            input="world",
            expected_trajectory=[ExpectedToolCall(name="search", args={"q": "world"})],
        ),
    ]
)


def test_run_eval_basic():
    with tempfile.TemporaryDirectory() as tmpdir:
        report = run_eval(DATASET, agent_fn=_mock_agent, data_dir=tmpdir)
        assert isinstance(report, RunReport)
        assert len(report.results) == 2
        assert "tool_correctness" in report.summary
        assert report.summary["tool_correctness"] == 1.0


def test_run_eval_with_judge():
    with tempfile.TemporaryDirectory() as tmpdir:
        report = run_eval(DATASET, agent_fn=_mock_agent, judge_fn=_mock_judge, data_dir=tmpdir)
        assert "task_success" in report.summary
        assert report.summary["task_success"] == 1.0



def test_run_eval_repeat():
    with tempfile.TemporaryDirectory() as tmpdir:
        report = run_eval(DATASET, agent_fn=_mock_agent, repeat=3, data_dir=tmpdir)
        assert report.reliability["pass_rate"]["tool_correctness"] == 1.0
        assert report.reliability["consistency"]["tool_correctness"] == 1.0
