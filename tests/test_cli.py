import json
import os
import tempfile
from unittest.mock import patch
from nanoevals.cli import main


def test_cli_gate_fail():
    with tempfile.TemporaryDirectory() as tmpdir:
        run_id = "test_run"
        run_dir = os.path.join(tmpdir, run_id)
        os.makedirs(run_dir)
        report = {
            "run_id": run_id,
            "timestamp": "2026-01-01",
            "results": [],
            "summary": {"tool_correctness": 0.1},
            "reliability": {},
        }
        with open(os.path.join(run_dir, "report.json"), "w") as f:
            json.dump(report, f)

        with patch("sys.argv", ["nanoeval", "gate", "--run-id", run_id, "--data-dir", tmpdir]):
            result = main()
            assert result == 1
