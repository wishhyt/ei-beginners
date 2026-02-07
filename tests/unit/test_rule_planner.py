from __future__ import annotations

from ei_beginners.embodied.models import TaskSpec
from ei_beginners.embodied.planning.rule_planner import RulePlanner


def test_rule_planner_is_deterministic_for_sample_prompt() -> None:
    task = TaskSpec(
        identifier="assembling_gift_baskets_demo_0001",
        natural_language_description="Put a candle_0 and a cookie_0 into basket_0.",
        scene_objects=["basket_0", "cookie_0", "candle_0"],
    )
    planner = RulePlanner()

    plan_a = planner.plan(task)
    plan_b = planner.plan(task)

    assert [(s.action, s.object, s.target) for s in plan_a.steps] == [
        ("GRASP", "candle_0", None),
        ("PLACE_INSIDE", "candle_0", "basket_0"),
        ("GRASP", "cookie_0", None),
        ("PLACE_INSIDE", "cookie_0", "basket_0"),
    ]
    assert [(s.action, s.object, s.target) for s in plan_a.steps] == [
        (s.action, s.object, s.target) for s in plan_b.steps
    ]
