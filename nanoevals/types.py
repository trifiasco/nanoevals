from typing import Any
from pydantic import BaseModel, Field


class ToolCall(BaseModel):
    name: str
    args: dict[str, Any] = Field(default_factory=dict)


class UsageStats(BaseModel):
    latency_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0


class Trace(BaseModel):
    input: str
    output: str
    tool_calls: list[ToolCall] = Field(default_factory=list)
    usage: UsageStats = Field(default_factory=UsageStats)


class EvalResult(BaseModel):
    metric: str
    score: float
    passed: bool
    comments: str = ""
