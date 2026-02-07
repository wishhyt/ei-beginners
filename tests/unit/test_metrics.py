from __future__ import annotations

from ei_beginners.embodied.evaluation.metrics import evaluate
from ei_beginners.embodied.models import ExecutionEvent, ExecutionTrace, TaskSpec


def test_metrics_partial_success_and_error_taxonomy() -> None:
    task = TaskSpec(
        identifier="task_partial",
        natural_language_description="Put candle_0 and cookie_0 into basket_0.",
        scene_objects=["basket_0", "candle_0", "cookie_0"],
    )

    trace = ExecutionTrace(
        identifier="task_partial",
        planner="rule",
        start_time="2026-01-01T00:00:00+00:00",
        end_time="2026-01-01T00:00:01+00:00",
        events=[
            ExecutionEvent(index=0, action="GRASP", object="candle_0", target=None, success=True),
            ExecutionEvent(
                index=1, action="PLACE_INSIDE", object="candle_0", target="basket_0", success=True
            ),
            ExecutionEvent(
                index=2,
                action="GRASP",
                object="cookie_0",
                target=None,
                success=False,
                error="not_grasped",
            ),
            ExecutionEvent(
                index=3,
                action="PLACE_INSIDE",
                object="cookie_0",
                target="basket_0",
                success=False,
                error="not_grasped",
            ),
        ],
        final_state={"container_contents": {"basket_0": ["candle_0"]}},
    )

    report = evaluate(task, trace)

    assert 0.0 < report.metrics.partial_success < 1.0
    assert report.metrics.error_taxonomy["not_grasped"] == 2
    assert report.metrics.goal_satisfaction == 0.5
