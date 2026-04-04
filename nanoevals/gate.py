DEFAULT_THRESHOLDS = {
    "tool_correctness": 0.90,
    "trajectory_match": 0.75,
    "step_efficiency": 0.70,
}


def check(report: dict, thresholds: dict = DEFAULT_THRESHOLDS) -> bool:
    failures = {
        k: (report[k], v)
        for k, v in thresholds.items()
        if k in report and report[k] < v
    }
    if failures:
        print("Deployment BLOCKED:")
        for m, (actual, required) in failures.items():
            print(f"  {m}: {actual:.3f} < {required:.3f}")
        return False
    return True
