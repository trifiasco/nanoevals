from typing import Any, Literal
from pathlib import Path
from pydantic import BaseModel, Field
import yaml


MatchMode = Literal["exact", "fuzzy"]


class ExpectedToolCall(BaseModel):
    name: str
    args: dict[str, Any] = Field(default_factory=dict)
    match_mode: MatchMode = "exact"


class AgentTestCase(BaseModel):
    input: str
    expected_trajectory: list[ExpectedToolCall] = Field(default_factory=list)
    success_criteria: list[str] = Field(default_factory=list)
    reference_output: str | None = None


class DatasetMeta(BaseModel):
    version: str = "0.0.0"
    description: str = ""


class AgentGoldenDataset(BaseModel):
    metadata: DatasetMeta = Field(default_factory=DatasetMeta)
    test_cases: list[AgentTestCase] = Field(default_factory=list)


class JudgeTestCase(BaseModel):
    metric: str
    trace: dict[str, Any]
    human_label: Literal["PASS", "FAIL"]
    notes: str = ""


class JudgeGoldenDataset(BaseModel):
    metadata: DatasetMeta = Field(default_factory=DatasetMeta)
    test_cases: list[JudgeTestCase] = Field(default_factory=list)


def load_agent_dataset(path: str) -> AgentGoldenDataset:
    return AgentGoldenDataset.model_validate(yaml.safe_load(Path(path).read_text()))


def load_judge_dataset(path: str) -> JudgeGoldenDataset:
    return JudgeGoldenDataset.model_validate(yaml.safe_load(Path(path).read_text()))


def save_agent_dataset(path: str, dataset: AgentGoldenDataset) -> None:
    Path(path).write_text(yaml.dump(dataset.model_dump(), default_flow_style=False))


def save_judge_dataset(path: str, dataset: JudgeGoldenDataset) -> None:
    Path(path).write_text(yaml.dump(dataset.model_dump(), default_flow_style=False))
