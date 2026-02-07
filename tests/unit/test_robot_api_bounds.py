from __future__ import annotations

from ei_beginners.embodied.execution.robot_api import RobotAPI
from ei_beginners.embodied.execution.simulation import EmbodiedSimulation, SimulationConfig


def test_robot_api_bounds_reject_out_of_workspace_position() -> None:
    sim = EmbodiedSimulation(SimulationConfig(backend="dummy"))
    sim.reset(["cube_0", "tray_0"])
    api = RobotAPI(sim)

    assert api.is_position_in_bounds((0.5, 0.0, 0.2))
    assert not api.is_position_in_bounds((2.0, 0.0, 0.2))
    assert not api.is_position_in_bounds((-0.1, 0.0, 0.2))
