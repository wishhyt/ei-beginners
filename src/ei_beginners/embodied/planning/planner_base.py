"""Planner protocol."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ei_beginners.embodied.models import ActionPlan, TaskSpec


class Planner(ABC):
    """Planner interface."""

    @abstractmethod
    def plan(self, task: TaskSpec) -> ActionPlan:
        """Generate an action plan for a task."""
        raise NotImplementedError
