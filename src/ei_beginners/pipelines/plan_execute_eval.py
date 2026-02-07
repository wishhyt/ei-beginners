"""Plan -> Execute -> Evaluate pipeline."""

from __future__ import annotations

import json
from pathlib import Path

from ei_beginners.embodied.evaluation.metrics import evaluate
from ei_beginners.embodied.evaluation.reporting import save_report
from ei_beginners.embodied.execution.action_executor import ActionExecutor
from ei_beginners.embodied.execution.robot_api import RobotAPI
from ei_beginners.embodied.execution.simulation import EmbodiedSimulation, SimulationConfig
from ei_beginners.embodied.models import TaskSpec
from ei_beginners.embodied.planning.llm_planner import LLMPlanner
from ei_beginners.embodied.planning.rule_planner import RulePlanner


def run_plan_execute_eval(
    task_file: str | Path,
    planner_name: str = "rule",
    report_dir: str | Path = "reports",
    backend: str = "dummy",
) -> dict[str, str]:
    task_path = Path(task_file)
    task_payload = json.loads(task_path.read_text(encoding="utf-8"))
    task = TaskSpec.from_dict(task_payload)

    planner = RulePlanner() if planner_name == "rule" else LLMPlanner()
    plan = planner.plan(task)

    simulation = EmbodiedSimulation(SimulationConfig(use_gui=False, backend=backend))
    simulation.reset(task.scene_objects)
    robot_api = RobotAPI(simulation)
    executor = ActionExecutor(robot_api)

    trace = executor.execute(plan)
    report = evaluate(task, trace)
    json_path, md_path = save_report(report, report_dir)

    simulation.close()

    return {
        "task_id": task.identifier,
        "planner": plan.planner,
        "json_report": str(json_path),
        "markdown_report": str(md_path),
    }
