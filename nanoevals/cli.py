import argparse
import importlib
import json
import subprocess
import sys
from pathlib import Path

from nanoevals.gate import check, DEFAULT_THRESHOLDS

DEFAULT_DATA_DIR = str(Path(__file__).parent.parent / "data" / "runs")


def _import_callable(module_path: str):
    module_name, fn_name = module_path.rsplit(":", 1)
    module = importlib.import_module(module_name)
    return getattr(module, fn_name)


def cmd_run(args):
    from nanoevals.dataset import load_agent_dataset
    from nanoevals.runner import run_eval

    dataset = load_agent_dataset(args.dataset)
    agent_fn = _import_callable(args.agent)
    judge_fn = _import_callable(args.judge) if args.judge else None
    extra_metrics = [_import_callable(m.strip()) for m in args.metrics.split(",")] if args.metrics else None
    report = run_eval(
        dataset,
        agent_fn=agent_fn,
        judge_fn=judge_fn,
        extra_metrics=extra_metrics,
        run_id=args.run_id,
        repeat=args.repeat,
        data_dir=args.data_dir,
    )
    print(f"Run {report.run_id} complete.")
    for k, v in report.summary.items():
        print(f"  {k}: {v:.3f}")
    return 0


def cmd_app(args):
    app_path = Path(__file__).parent.parent / "app.py"
    cmd = ["streamlit", "run", str(app_path), "--", "--data-dir", args.data_dir]
    subprocess.run(cmd)
    return 0


def cmd_gate(args):
    report_path = Path(args.data_dir) / args.run_id / "report.json"
    report = json.loads(report_path.read_text())
    thresholds = DEFAULT_THRESHOLDS
    if args.thresholds:
        thresholds = json.loads(Path(args.thresholds).read_text())
    passed, failures = check(report["summary"], thresholds)
    if not passed:
        print("Deployment BLOCKED:")
        for m, (actual, required) in failures.items():
            print(f"  {m}: {actual:.3f} < {required:.3f}")
    return 0 if passed else 1


def main():
    parser = argparse.ArgumentParser(prog="nanoeval")
    sub = parser.add_subparsers(dest="command")

    run_p = sub.add_parser("run")
    run_p.add_argument("--dataset", required=True)
    run_p.add_argument("--agent", required=True, help="module:function")
    run_p.add_argument("--judge", help="module:function")
    run_p.add_argument("--metrics", help="comma-separated module:function paths")
    run_p.add_argument("--run-id")
    run_p.add_argument("--repeat", type=int, default=1)
    run_p.add_argument("--data-dir", default=DEFAULT_DATA_DIR)

    app_p = sub.add_parser("app")
    app_p.add_argument("--data-dir", default=DEFAULT_DATA_DIR)

    gate_p = sub.add_parser("gate")
    gate_p.add_argument("--run-id", required=True)
    gate_p.add_argument("--thresholds", help="path to JSON thresholds file")
    gate_p.add_argument("--data-dir", default=DEFAULT_DATA_DIR)

    args = parser.parse_args()
    commands = {"run": cmd_run, "app": cmd_app, "gate": cmd_gate}
    if args.command in commands:
        return commands[args.command](args)
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
