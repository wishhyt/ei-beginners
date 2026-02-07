"""Deterministic rule-based planner."""

from __future__ import annotations

import re

from ei_beginners.embodied.models import ActionPlan, ActionStep, TaskSpec
from ei_beginners.embodied.planning.planner_base import Planner


class RulePlanner(Planner):
    """Deterministic planner for baseline execution and tests."""

    def plan(self, task: TaskSpec) -> ActionPlan:
        container = self._find_container(task.scene_objects)
        movable_objects = [obj for obj in task.scene_objects if obj != container]
        selected_objects = self._extract_mentioned_objects(task, movable_objects)
        steps: list[ActionStep] = []
        for obj in selected_objects:
            steps.append(ActionStep(action="GRASP", object=obj))
            steps.append(ActionStep(action="PLACE_INSIDE", object=obj, target=container))

        plan = ActionPlan(identifier=task.identifier, planner="rule", steps=steps)
        plan.validate()
        return plan

    def _extract_mentioned_objects(self, task: TaskSpec, default: list[str]) -> list[str]:
        text = task.natural_language_description.lower()
        referenced = [obj for obj in default if obj.lower() in text]
        if referenced:
            return sorted(referenced)
        # fallback for noun-like object mentions in sentence
        nouns = re.findall(r"\b([a-z]+_\d+)\b", text)
        if nouns:
            return sorted([obj for obj in default if obj in nouns])
        return sorted(default)

    @staticmethod
    def _find_container(scene_objects: list[str]) -> str:
        for obj in sorted(scene_objects):
            lowered = obj.lower()
            if (
                lowered.startswith("basket")
                or lowered.startswith("tray")
                or lowered.startswith("bowl")
            ):
                return obj
        return sorted(scene_objects)[-1] if scene_objects else "container_0"
