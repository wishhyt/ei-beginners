from __future__ import annotations

import pytest

from ei_beginners.embodied.models import ActionPlan, ActionStep, ValidationError


def test_action_schema_validation_rejects_malformed_output() -> None:
    with pytest.raises(ValidationError):
        ActionStep(action="", object="cube_0").validate()

    with pytest.raises(ValidationError):
        ActionStep(action="GRASP", object="").validate()

    # malformed plan: step has invalid target type
    step = ActionStep(action="PLACE_INSIDE", object="cube_0", target=None)
    step.target = 123  # type: ignore[assignment]
    with pytest.raises(ValidationError):
        plan = ActionPlan(identifier="task_1", planner="rule", steps=[step])
        plan.validate()
