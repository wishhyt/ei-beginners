"""Action plan executor."""

from __future__ import annotations

from datetime import datetime, timezone

from ei_beginners.embodied.execution.robot_api import RobotAPI
from ei_beginners.embodied.models import ActionPlan, ExecutionEvent, ExecutionTrace


class ActionExecutor:
    """Executes symbolic action plans against RobotAPI."""

    def __init__(self, robot_api: RobotAPI) -> None:
        self.robot_api = robot_api

    def execute(self, plan: ActionPlan) -> ExecutionTrace:
        trace = ExecutionTrace.create(identifier=plan.identifier, planner=plan.planner)
        for idx, step in enumerate(plan.steps):
            success = False
            error: str | None = None
            action = step.action.upper()

            if action == "GRASP":
                success, error = self.robot_api.grasp(step.object)
            elif action == "PLACE_INSIDE":
                if not step.target:
                    success, error = False, "missing_target"
                else:
                    success, error = self.robot_api.place_inside(step.object, step.target)
            elif action == "MOVE_TO_POSITION":
                coords = step.metadata.get("position")
                if not isinstance(coords, (list, tuple)) or len(coords) != 3:
                    success, error = False, "missing_position"
                else:
                    success, error = self.robot_api.move_to_position(
                        step.object, float(coords[0]), float(coords[1]), float(coords[2])
                    )
            else:
                success, error = False, "unknown_action"

            trace.events.append(
                ExecutionEvent(
                    index=idx,
                    action=action,
                    object=step.object,
                    target=step.target,
                    success=success,
                    error=error,
                )
            )

        trace.end_time = datetime.now(timezone.utc).isoformat()
        trace.final_state = self.robot_api.get_state()
        return trace
