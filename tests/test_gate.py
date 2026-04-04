from nanoevals.gate import check, DEFAULT_THRESHOLDS


def test_gate_passes():
    report = {k: 1.0 for k in DEFAULT_THRESHOLDS}
    assert check(report) is True


def test_gate_fails(capsys):
    report = {"tool_correctness": 0.5, "trajectory_match": 1.0}
    thresholds = {"tool_correctness": 0.9, "trajectory_match": 0.75}
    assert check(report, thresholds) is False
    captured = capsys.readouterr()
    assert "tool_correctness" in captured.out
