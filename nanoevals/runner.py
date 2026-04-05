import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel, Field

from nanoevals.types import Trace, EvalResult
from nanoevals.dataset import AgentTestCase, AgentGoldenDataset
from nanoevals.metrics import tool_correctness, trajectory_match, step_efficiency, reference_match

DEFAULT_METRICS = [tool_correctness, trajectory_match, step_efficiency, reference_match]
class RunReport(BaseModel):
    run_id: str
    timestamp: str
    results: list[dict[str, Any]]
    summary: dict[str, float] = Field(default_factory=dict)
    reliability: dict[str, Any] = Field(default_factory=dict)


def _save_run(data_dir: str, run_id: str, report: RunReport, traces: list[Trace]) -> None:
    run_dir = Path(data_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "report.json").write_text(report.model_dump_json(indent=2))
    traces_dir = run_dir / "traces"
    traces_dir.mkdir(exist_ok=True)
    for i, trace in enumerate(traces):
        (traces_dir / f"{i}.json").write_text(trace.model_dump_json(indent=2))


def run_eval(
    dataset: AgentGoldenDataset,
    agent_fn: Callable[[str], Trace],
    judge_fn: Callable[[Trace, AgentTestCase], list[EvalResult]] | None = None,
    extra_metrics: list | None = None,
    run_id: str | None = None,
    repeat: int = 1,
    data_dir: str = "data/runs",
) -> RunReport:
    run_id = run_id or f"{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
    metrics = DEFAULT_METRICS + (extra_metrics or [])

    all_results: list[dict] = []
    all_traces: list[Trace] = []
    all_scores: dict[str, list[float]] = defaultdict(list)

    for rep in range(repeat):
        for i, test_case in enumerate(dataset.test_cases):
            try:
                trace = agent_fn(test_case.input)
            except Exception as exc:
                trace = Trace(output=f"ERROR: {exc}")

            if not trace.input:
                trace = trace.model_copy(update={"input": test_case.input})
            all_traces.append(trace)
            evals = [m(trace, test_case) for m in metrics]
            if judge_fn:
                evals.extend(judge_fn(trace, test_case))

            for e in evals:
                all_scores[e.metric].append(e.score)

            if rep == 0:
                all_results.append({
                    "input": test_case.input,
                    "output": trace.output,
                    "tool_calls": [tc.model_dump() for tc in trace.tool_calls],
                    "usage": trace.usage.model_dump(),
                    "evals": [e.model_dump() for e in evals],
                })

    summary = {k: sum(v) / len(v) for k, v in all_scores.items()}

    first_run_traces = all_traces[: len(dataset.test_cases)]
    total_tokens = sum(t.usage.input_tokens + t.usage.output_tokens for t in all_traces)
    total_cost = sum(t.usage.cost for t in all_traces)
    avg_latency = sum(t.usage.latency_ms for t in all_traces) / len(all_traces) if all_traces else 0

    reliability: dict[str, Any] = {
        "avg_latency_ms": avg_latency,
        "total_tokens": total_tokens,
        "total_cost": total_cost,
    }

    if repeat > 1:
        n_cases = len(dataset.test_cases)
        pass_rate = {}
        consistency = {}
        for metric_name, scores in all_scores.items():
            chunks = [scores[i * n_cases : (i + 1) * n_cases] for i in range(repeat)]
            per_case_pass = [sum(1 for c in chunk if c >= 0.7) / n_cases for chunk in chunks]
            pass_rate[metric_name] = sum(per_case_pass) / len(per_case_pass)
            per_case_results = list(zip(*chunks))
            consistent = sum(1 for case_scores in per_case_results if len(set(case_scores)) == 1)
            consistency[metric_name] = consistent / n_cases
        reliability["pass_rate"] = pass_rate
        reliability["consistency"] = consistency

    report = RunReport(
        run_id=run_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        results=all_results,
        summary=summary,
        reliability=reliability,
    )
    _save_run(data_dir, run_id, report, first_run_traces)
    return report
