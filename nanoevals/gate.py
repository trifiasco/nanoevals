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
