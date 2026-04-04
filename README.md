# nanoevals

A minimal eval library for AI agents. Illustrates the end to end flow of running evals, managing datasets, metrics. Comes with a CLI and a simple streamlit dashboard.

<img src="screenshots/Screenshot_Run_Eval.png" width="30%"> <img src="screenshots/Screenshot_Reports.png" width="30%"> <img src="screenshots/Screenshot_Dataset_Editor.png" width="30%">

## Install

```bash
uv pip install -e .
uv pip install -e ".[app]"  # for streamlit dashboard
```

## Quick Start

Run the streamlit app if you want a dashboard to visualize eval runs, results, add/edit datasets.

```
nanoevals app
```
Or, the CLI

```bash
nanoevals run --dataset examples/datasets/agent_golden.yaml --agent examples.mock_agent:mock_agent
nanoevals gate --run-id <run_id>
```

It expects the agent and judge as a function. So you can easily hook up your agent and judge and it should run against them. See `examples/run_eval.py` for reference.

## Structure

| Module | Lines | Purpose |
|--------|------:|---------|
| `nanoevals/types.py` | 28 | Trace, ToolCall, EvalResult, UsageStats |
| `nanoevals/dataset.py` | 59 | Dataset schemas + YAML load/save |
| `nanoevals/metrics.py` | 65 | tool_correctness, trajectory_match, step_efficiency |
| `nanoevals/runner.py` | 103 | Eval runner with reliability stats |
| `nanoevals/gate.py` | 14 | CI deployment gate |
| `nanoevals/cli.py` | 90 | CLI entry points |
| `app.py` | 198 | Streamlit dashboard |
| **Total** | **557** | |
