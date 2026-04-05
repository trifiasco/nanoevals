from nanoevals.types import Trace, EvalResult
from nanoevals.dataset import AgentTestCase


def response_verbosity(trace: Trace, test_case: AgentTestCase) -> EvalResult:
    if not test_case.reference_output:
        return EvalResult(metric="response_verbosity", score=1.0, passed=True)
    ratio = len(trace.output) / len(test_case.reference_output)
    score = 1.0 - abs(1.0 - ratio)
    score = max(score, 0.0)
    return EvalResult(
        metric="response_verbosity",
        score=score,
        passed=score >= 0.5,
        comments=f"actual/reference length ratio: {ratio:.2f}",
    )
