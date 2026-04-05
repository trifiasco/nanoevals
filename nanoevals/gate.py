from typing import Callable

from nanoevals.types import Trace, EvalResult
from nanoevals.dataset import JudgeGoldenDataset, AgentTestCase


DEFAULT_THRESHOLDS = {
    "tool_correctness": 0.90,
    "trajectory_match": 0.75,
    "step_efficiency": 0.70,
}


def check(report: dict, thresholds: dict = DEFAULT_THRESHOLDS) -> tuple[bool, dict]:
    failures = {
        k: (report[k], v)
        for k, v in thresholds.items()
        if k in report and report[k] < v
    }
    return (len(failures) == 0, failures)


def calibrate_judge(
    dataset: JudgeGoldenDataset,
    judge_fn: Callable[[Trace, AgentTestCase], list[EvalResult]],
) -> dict:
    tp = fp = tn = fn = 0
    for tc in dataset.test_cases:
        trace = Trace(output=tc.trace.get("output", ""), input=tc.trace.get("input", ""))
        case = AgentTestCase(input=trace.input)
        results = judge_fn(trace, case)
        judge_passed = any(r.passed for r in results if r.metric == tc.metric)
        human_passed = tc.human_label == "PASS"

        if judge_passed and human_passed:
            tp += 1
        elif judge_passed and not human_passed:
            fp += 1
        elif not judge_passed and human_passed:
            fn += 1
        else:
            tn += 1

    total = tp + fp + tn + fn
    return {
        "accuracy": (tp + tn) / total if total else 0.0,
        "tpr": tp / (tp + fn) if (tp + fn) else 0.0,
        "tnr": tn / (tn + fp) if (tn + fp) else 0.0,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }
