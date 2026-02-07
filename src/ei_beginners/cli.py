"""Command line interface for EI Beginners."""

from __future__ import annotations

import argparse
from pathlib import Path

from ei_beginners.pipelines.plan_execute_eval import run_plan_execute_eval
from ei_beginners.rl.bc.train import train_bc
from ei_beginners.rl.dqn.train import train_dqn
from ei_beginners.rl.ppo.train import train_ppo
from ei_beginners.rl.sac.train import train_sac


def _handle_plan_exec_eval(args: argparse.Namespace) -> int:
    result = run_plan_execute_eval(
        task_file=args.task_file,
        planner_name=args.planner,
        report_dir=args.report_dir,
        backend=args.backend,
    )
    print(result["json_report"])
    print(result["markdown_report"])
    return 0


def _handle_train_rl(args: argparse.Namespace) -> int:
    algo = args.algo.lower()
    if algo == "dqn":
        path = train_dqn(args.env, args.steps, args.output, seed=args.seed)
    elif algo == "ppo":
        path = train_ppo(args.env, args.steps, args.output, seed=args.seed)
    elif algo == "sac":
        path = train_sac(args.env, args.steps, args.output, seed=args.seed)
    else:
        raise ValueError(f"Unsupported algorithm: {args.algo}")

    print(path)
    return 0


def _handle_train_bc(args: argparse.Namespace) -> int:
    path = train_bc(dataset=args.dataset, output_dir=args.output, epochs=args.epochs)
    print(path)
    return 0


def _handle_journey_show(_: argparse.Namespace) -> int:
    path = Path("learning_journey/LEARNING_PATH.md")
    if not path.exists():
        raise FileNotFoundError("learning_journey/LEARNING_PATH.md not found")
    print(path.resolve())
    print(path.read_text(encoding="utf-8"))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ei", description="EI Beginners unified CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan_parser = subparsers.add_parser(
        "plan-exec-eval", help="Run plan -> execute -> evaluate pipeline"
    )
    plan_parser.add_argument("--task-file", required=True)
    plan_parser.add_argument("--planner", choices=["rule", "llm"], default="rule")
    plan_parser.add_argument("--report-dir", default="reports")
    plan_parser.add_argument("--backend", choices=["dummy", "pybullet"], default="dummy")
    plan_parser.set_defaults(func=_handle_plan_exec_eval)

    rl_parser = subparsers.add_parser("train-rl", help="Train RL baseline model")
    rl_parser.add_argument("--algo", choices=["dqn", "ppo", "sac"], required=True)
    rl_parser.add_argument("--env", default="CartPole-v1")
    rl_parser.add_argument("--steps", type=int, default=1000)
    rl_parser.add_argument("--output", default="outputs/rl")
    rl_parser.add_argument("--seed", type=int, default=0)
    rl_parser.set_defaults(func=_handle_train_rl)

    bc_parser = subparsers.add_parser("train-bc", help="Train behavior cloning baseline")
    bc_parser.add_argument("--dataset", required=True)
    bc_parser.add_argument("--epochs", type=int, default=1)
    bc_parser.add_argument("--output", default="outputs/rl/bc")
    bc_parser.set_defaults(func=_handle_train_bc)

    journey_parser = subparsers.add_parser("journey", help="Learning journey commands")
    journey_subparsers = journey_parser.add_subparsers(dest="journey_cmd", required=True)
    journey_show = journey_subparsers.add_parser("show", help="Show learning path")
    journey_show.set_defaults(func=_handle_journey_show)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
