"""Robot API over simulation state."""

from __future__ import annotations

from typing import Any

from ei_beginners.embodied.execution.simulation import EmbodiedSimulation


class RobotAPI:
    """Robot control abstraction used by action executor."""

    def __init__(self, simulation: EmbodiedSimulation) -> None:
        self.simulation = simulation
        self.holding: str | None = None

    def detect_objects(self, object_type: str | None = None) -> list[str]:
        objects = sorted(self.simulation.object_positions.keys())
        if object_type is None:
            return objects
        lowered = object_type.lower()
        if lowered.endswith("s"):
            lowered = lowered[:-1]
        return [obj for obj in objects if lowered in obj.lower()]

    def grasp(self, object_name: str) -> tuple[bool, str | None]:
        if object_name not in self.simulation.object_positions:
            return False, "object_not_found"
        self.holding = object_name
        return True, None

    def place_inside(self, object_name: str, container_name: str) -> tuple[bool, str | None]:
        if self.holding != object_name:
            return False, "not_grasped"
        ok = self.simulation.place_inside(object_name, container_name)
        if not ok:
            return False, "placement_failed"
        self.holding = None
        return True, None

    def move_to_position(
        self, object_name: str, x: float, y: float, z: float
    ) -> tuple[bool, str | None]:
        ok = self.simulation.move_object(object_name, (x, y, z))
        if not ok:
            return False, "object_not_found"
        return True, None

    def is_empty(self, container_name: str) -> bool:
        return len(self.simulation.container_contents.get(container_name, [])) == 0

    def is_position_in_bounds(self, position: tuple[float, float, float]) -> bool:
        x, y, z = position
        return 0.0 <= x <= 1.5 and -1.0 <= y <= 1.0 and 0.0 <= z <= 1.5

    def get_state(self) -> dict[str, Any]:
        return self.simulation.snapshot()
