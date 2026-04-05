import json
import sys
from pathlib import Path

import streamlit as st
import yaml

from nanoevals.dataset import (
    AgentGoldenDataset,
    AgentTestCase,
    ExpectedToolCall,
    load_agent_dataset,
    save_agent_dataset,
)
from nanoevals.gate import check, DEFAULT_THRESHOLDS
from nanoevals.runner import run_eval
from nanoevals.cli import _import_callable


DATA_DIR = "data/runs"
for i, arg in enumerate(sys.argv):
    if arg == "--data-dir" and i + 1 < len(sys.argv):
        DATA_DIR = sys.argv[i + 1]

st.set_page_config(page_title="nanoEvals", layout="wide")
st.title("nanoEvals")

tab_reports, tab_agent_editor, tab_run = st.tabs(
    ["Previous Run Reports", "Dataset Editor", "Run Eval"]
)


def list_runs():
    runs_path = Path(DATA_DIR)
    if not runs_path.exists():
        return []
    return sorted(
        [d.name for d in runs_path.iterdir() if d.is_dir() and (d / "report.json").exists()],
        reverse=True,
    )


def load_report(run_id: str) -> dict:
    return json.loads((Path(DATA_DIR) / run_id / "report.json").read_text())


with tab_reports:
    runs = list_runs()
    if not runs:
        st.info("No runs found. Run an eval first.")
    else:
        selected = st.selectbox("Select run", runs)
        report = load_report(selected)

        st.subheader(f"Run: {report['run_id']}")
        st.text(f"Timestamp: {report['timestamp']}")

        gate_passed, gate_failures = check(report["summary"], DEFAULT_THRESHOLDS)
        if gate_passed:
            st.success("CI Gate: PASS")
        else:
            lines = [f"- {m}: {a:.3f} < {r:.3f}" for m, (a, r) in gate_failures.items()]
            st.error("CI Gate: FAIL\n" + "\n".join(lines))

        st.subheader("Summary")
        cols = st.columns(len(report["summary"]))
        for col, (metric, score) in zip(cols, report["summary"].items()):
            col.metric(metric, f"{score:.3f}")

        if report.get("reliability"):
            st.subheader("Reliability")
            rel = report["reliability"]
            r_cols = st.columns(3)
            r_cols[0].metric("Avg Latency", f"{rel.get('avg_latency_ms', 0):.0f}ms")
            r_cols[1].metric("Total Tokens", rel.get("total_tokens", 0))
            r_cols[2].metric("Total Cost", f"${rel.get('total_cost', 0):.4f}")

            if "pass_rate" in rel:
                st.write("**Pass Rate:**", rel["pass_rate"])
            if "consistency" in rel:
                st.write("**Consistency:**", rel["consistency"])

        st.subheader("Results")
        for i, result in enumerate(report["results"]):
            with st.expander(f"Case {i + 1}: {result['input'][:80]}"):
                st.write("**Output:**", result["output"])
                st.write("**Tool Calls:**")
                st.json(result["tool_calls"])
                if result.get("usage"):
                    st.write("**Usage:**")
                    st.json(result["usage"])
                st.write("**Evals:**")
                for ev in result.get("evals", []):
                    status = "PASS" if ev["passed"] else "FAIL"
                    st.write(f"- {ev['metric']}: {ev['score']:.3f} ({status}) — {ev['comments']}")


with tab_agent_editor:
    st.subheader("Agent Dataset Editor")
    agent_uploaded = st.file_uploader("Load dataset YAML", type=["yaml", "yml"], key="agent_upload")
    agent_save_path = st.text_input("Save path", "examples/datasets/agent_golden.yaml", key="agent_path")

    if agent_uploaded:
        st.session_state.agent_dataset = AgentGoldenDataset.model_validate(
            yaml.safe_load(agent_uploaded.getvalue())
        )
    elif "agent_dataset" not in st.session_state:
        if Path(agent_save_path).exists():
            st.session_state.agent_dataset = load_agent_dataset(agent_save_path)
        else:
            st.session_state.agent_dataset = AgentGoldenDataset()

    dataset = st.session_state.agent_dataset
    st.write(f"**{len(dataset.test_cases)} test cases**")

    to_delete = None
    for i, tc in enumerate(dataset.test_cases):
        with st.expander(f"Case {i + 1}: {tc.input[:60]}"):
            new_input = st.text_area(f"Input##agent{i}", tc.input)
            new_ref = st.text_area(f"Reference Output##agent{i}", tc.reference_output or "")
            trajectory_data = [
                {"name": t.name, "args": t.args, "match_mode": t.match_mode}
                for t in tc.expected_trajectory
            ]
            trajectory_yaml = yaml.dump(
                trajectory_data, default_flow_style=False, sort_keys=False,
            ) if tc.expected_trajectory else ""
            new_trajectory_yaml = st.text_area(
                f"Expected Trajectory (YAML)##agent{i}",
                trajectory_yaml,
                help="List of tool calls: name, args, match_mode (exact/fuzzy)",
            )
            new_criteria = st.text_area(
                f"Success Criteria (one per line)##agent{i}",
                "\n".join(tc.success_criteria),
            )

            new_trajectory = tc.expected_trajectory
            if new_trajectory_yaml.strip():
                parsed = yaml.safe_load(new_trajectory_yaml)
                if isinstance(parsed, list):
                    new_trajectory = [ExpectedToolCall.model_validate(t) for t in parsed]

            dataset.test_cases[i] = tc.model_copy(update={
                "input": new_input,
                "reference_output": new_ref or None,
                "expected_trajectory": new_trajectory,
                "success_criteria": [s.strip() for s in new_criteria.split("\n") if s.strip()],
            })

            if st.button(f"Delete##agent{i}"):
                to_delete = i

    if to_delete is not None:
        dataset.test_cases.pop(to_delete)
        st.rerun()

    if st.button("Add Test Case", key="add_agent"):
        dataset.test_cases.append(AgentTestCase(input="New test case"))
        st.rerun()

    if st.button("Save Dataset", key="save_agent"):
        save_agent_dataset(agent_save_path, dataset)
        st.success(f"Saved to {agent_save_path}")


with tab_run:
    st.subheader("Run Eval")
    dataset_path = st.text_input("Dataset path", "examples/datasets/agent_golden.yaml")
    agent_module = st.text_input("Agent module (module:function)", "examples.mock_agent:mock_agent")
    judge_module = st.text_input("Judge module (module:function, optional)", "examples.judge:simple_judge")
    metrics_input = st.text_input("Extra metrics (comma-separated module:function)", "examples.custom_metrics:response_verbosity")
    repeat = st.number_input("Repeat", min_value=1, max_value=100, value=1)

    if st.button("Run"):
        try:
            ds = load_agent_dataset(dataset_path)
            agent_fn = _import_callable(agent_module)
            judge_fn = _import_callable(judge_module) if judge_module else None
            extra_metrics = [_import_callable(m.strip()) for m in metrics_input.split(",") if m.strip()] or None
        except Exception as e:
            st.error(f"Failed to load: {e}")
            st.stop()

        with st.spinner("Running eval..."):
            report = run_eval(ds, agent_fn=agent_fn, judge_fn=judge_fn, extra_metrics=extra_metrics, repeat=repeat, data_dir=DATA_DIR)

        st.success(f"Run {report.run_id} complete! Reloading...")
        st.rerun()
