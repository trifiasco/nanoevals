from nanoevals.gate import check, DEFAULT_THRESHOLDS


def test_gate_passes():
    report = {k: 1.0 for k in DEFAULT_THRESHOLDS}
    passed, failures = check(report)
    assert passed is True
    assert failures == {}


def test_gate_fails():
    report = {"tool_correctness": 0.5, "trajectory_match": 1.0}
    thresholds = {"tool_correctness": 0.9, "trajectory_match": 0.75}
    passed, failures = check(report, thresholds)
    assert passed is False
    assert "tool_correctness" in failures
