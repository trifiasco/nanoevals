import os
import tempfile
from nanoevals.dataset import (
    AgentTestCase,
    AgentGoldenDataset,
    ExpectedToolCall,
    load_agent_dataset,
    load_judge_dataset,
    save_agent_dataset,
)

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


def test_load_agent_dataset():
    ds = load_agent_dataset(os.path.join(FIXTURES, "agent_golden.yaml"))
    assert len(ds.test_cases) == 2
    assert ds.test_cases[0].input == "What is the weather in Tokyo?"
    assert ds.test_cases[0].expected_trajectory[0].name == "get_weather"
    assert ds.test_cases[0].expected_trajectory[0].match_mode == "exact"


def test_load_judge_dataset():
    ds = load_judge_dataset(os.path.join(FIXTURES, "judge_golden.yaml"))
    assert len(ds.test_cases) == 2
    assert ds.test_cases[0].human_label == "PASS"
    assert ds.test_cases[1].human_label == "FAIL"


def test_save_agent_dataset():
    ds = AgentGoldenDataset(
        test_cases=[
            AgentTestCase(
                input="test",
                expected_trajectory=[ExpectedToolCall(name="fn", args={"a": 1})],
            )
        ]
    )
    with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
        path = f.name
    try:
        save_agent_dataset(path, ds)
        loaded = load_agent_dataset(path)
        assert len(loaded.test_cases) == 1
        assert loaded.test_cases[0].input == "test"
    finally:
        os.unlink(path)
