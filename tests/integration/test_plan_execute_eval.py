from __future__ import annotations

from pathlib import Path

from ei_beginners.pipelines.plan_execute_eval import run_plan_execute_eval

ROOT = Path(__file__).resolve().parents[2]


def test_pipeline_end_to_end_with_sample_task(tmp_path: Path) -> None:
    task_file = ROOT / "learning_journey" / "phase_04_eai_evaluation" / "samples" / "prompt.json"
    result = run_plan_execute_eval(
        task_file=task_file, planner_name="rule", report_dir=tmp_path, backend="dummy"
    )

    json_report = Path(result["json_report"])
    md_report = Path(result["markdown_report"])

    assert json_report.exists()
    assert md_report.exists()
    assert "_report.json" in json_report.name
