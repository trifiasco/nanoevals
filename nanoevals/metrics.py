from nanoevals.types import Trace, EvalResult
from nanoevals.dataset import AgentTestCase


def tool_correctness(trace: Trace, test_case: AgentTestCase) -> EvalResult:
    expected = test_case.expected_trajectory
    if not expected:
        return EvalResult(metric="tool_correctness", score=1.0, passed=True)

    def _match(actual_call, exp_call):
        if actual_call.name != exp_call.name:
            return False
        if exp_call.match_mode == "fuzzy":
            return all(item in actual_call.args.items() for item in exp_call.args.items())
        return actual_call.args == exp_call.args

    matches = sum(
        any(_match(ac, ec) for ac in trace.tool_calls) for ec in expected
    )
    score = matches / len(expected)
    return EvalResult(
        metric="tool_correctness",
        score=score,
        passed=score == 1.0,
        comments=f"{matches}/{len(expected)} matched",
    )


def trajectory_match(trace: Trace, test_case: AgentTestCase) -> EvalResult:
    expected = test_case.expected_trajectory
    if not expected:
        return EvalResult(metric="trajectory_match", score=1.0, passed=True)

    actual_names = [tc.name for tc in trace.tool_calls]
    expected_names = [tc.name for tc in expected]

    i = 0
    lcs = 0
    for name in actual_names:
        if i < len(expected_names) and name == expected_names[i]:
            lcs += 1
            i += 1

    score = lcs / len(expected_names)
    return EvalResult(
        metric="trajectory_match",
        score=score,
        passed=score == 1.0,
        comments=f"LCS {lcs}/{len(expected_names)}",
    )


def step_efficiency(trace: Trace, test_case: AgentTestCase) -> EvalResult:
    actual = len(trace.tool_calls)
    expected = len(test_case.expected_trajectory)
    if actual == 0 and expected == 0:
        return EvalResult(metric="step_efficiency", score=1.0, passed=True)

    score = min(actual, expected) / max(actual, expected)
    return EvalResult(
        metric="step_efficiency",
        score=score,
        passed=score >= 0.7,
        comments=f"{actual} actual / {expected} expected",
    )
