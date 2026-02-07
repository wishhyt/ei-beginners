"""Lightweight embodied simulation backend with optional PyBullet support."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:
    import pybullet as p
    import pybullet_data
except ImportError:  # pragma: no cover - optional dependency
    p = None
    pybullet_data = None


@dataclass
class SimulationConfig:
    use_gui: bool = False
    backend: str = "dummy"  # dummy | pybullet


class EmbodiedSimulation:
    """Scene state store with optional PyBullet initialization."""

    def __init__(self, config: SimulationConfig | None = None) -> None:
        self.config = config or SimulationConfig()
        self.object_positions: dict[str, tuple[float, float, float]] = {}
        self.container_contents: dict[str, list[str]] = {}
        self._connected = False
        self._client_id: int | None = None

    def reset(self, scene_objects: list[str]) -> None:
        self.object_positions = {}
        self.container_contents = {}
        for idx, obj in enumerate(sorted(scene_objects)):
            self.object_positions[obj] = (0.4 + idx * 0.05, 0.0, 0.02)
            self.container_contents[obj] = []

        if self.config.backend == "pybullet" and p is not None:
            self._connect_pybullet()

    def _connect_pybullet(self) -> None:
        if self._connected:
            return
        mode = p.GUI if self.config.use_gui else p.DIRECT
        self._client_id = p.connect(mode)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.resetSimulation()
        p.setGravity(0, 0, -9.8)
        p.loadURDF("plane.urdf")
        self._connected = True

    def move_object(self, object_name: str, position: tuple[float, float, float]) -> bool:
        if object_name not in self.object_positions:
            return False
        self.object_positions[object_name] = position
        return True

    def place_inside(self, object_name: str, container_name: str) -> bool:
        if object_name not in self.object_positions:
            return False
        if container_name not in self.object_positions:
            return False
        self.container_contents.setdefault(container_name, [])
        if object_name not in self.container_contents[container_name]:
            self.container_contents[container_name].append(object_name)
        container_pos = self.object_positions[container_name]
        self.object_positions[object_name] = (
            container_pos[0],
            container_pos[1],
            container_pos[2] + 0.05,
        )
        return True

    def get_object_position(self, object_name: str) -> tuple[float, float, float] | None:
        return self.object_positions.get(object_name)

    def is_inside(self, object_name: str, container_name: str) -> bool:
        return object_name in self.container_contents.get(container_name, [])

    def snapshot(self) -> dict[str, Any]:
        return {
            "object_positions": self.object_positions.copy(),
            "container_contents": {k: v[:] for k, v in self.container_contents.items()},
        }

    def close(self) -> None:
        if self._connected and p is not None and self._client_id is not None:
            p.disconnect(self._client_id)
        self._connected = False
        self._client_id = None
