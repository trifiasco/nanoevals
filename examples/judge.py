from nanoevals.types import Trace, EvalResult
from nanoevals.dataset import AgentTestCase


def simple_judge(trace: Trace, test_case: AgentTestCase) -> list[EvalResult]:
    if not test_case.success_criteria:
        return []
    hits = sum(1 for c in test_case.success_criteria if c.lower() in trace.output.lower())
    score = hits / len(test_case.success_criteria)
    return [EvalResult(
        metric="task_success",
        score=score,
        passed=score >= 0.5,
        comments=f"{hits}/{len(test_case.success_criteria)} criteria met",
    )]
