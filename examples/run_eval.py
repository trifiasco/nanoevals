from pathlib import Path

from nanoevals.dataset import load_agent_dataset
from nanoevals.runner import run_eval
from nanoevals.gate import check
from mock_agent import mock_agent
from judge import simple_judge
from custom_metrics import response_verbosity

dataset = load_agent_dataset(str(Path(__file__).parent / "datasets" / "agent_golden.yaml"))

report = run_eval(
    dataset,
    agent_fn=mock_agent,
    judge_fn=simple_judge,
    extra_metrics=[response_verbosity],
)

print(f"Run: {report.run_id}")
print(f"Timestamp: {report.timestamp}")
print("\nMetric Scores:")
for k, v in report.summary.items():
    print(f"  {k}: {v:.3f}")

print("\nReliability:")
print(f"  Avg latency: {report.reliability['avg_latency_ms']:.0f}ms")
print(f"  Total tokens: {report.reliability['total_tokens']}")
print(f"  Total cost: ${report.reliability['total_cost']:.4f}")

passed, failures = check(report.summary)
print(f"\nCI Gate: {'PASS' if passed else 'FAIL'}")
for m, (actual, required) in failures.items():
    print(f"  {m}: {actual:.3f} < {required:.3f}")
