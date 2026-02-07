"""Typed models for planning, execution, and evaluation."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


class ValidationError(ValueError):
    """Raised when model validation fails."""


@dataclass
class TaskSpec:
    """Input task specification for planning and evaluation."""

    identifier: str
    natural_language_description: str
    scene_objects: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TaskSpec:
        identifier = str(payload.get("identifier") or payload.get("task_id") or "unknown_task")
        description = payload.get("natural_language_description")
        if description is None:
            description = payload.get("llm_prompt", "")

        scene_objects_raw = payload.get("scene_objects")
        if scene_objects_raw is None:
            scene_objects_raw = cls._parse_scene_objects(str(description))

        scene_objects = [str(item) for item in scene_objects_raw]
        metadata = {
            k: v
            for k, v in payload.items()
            if k
            not in {
                "identifier",
                "task_id",
                "natural_language_description",
                "llm_prompt",
                "scene_objects",
            }
        }

        task = cls(
            identifier=identifier,
            natural_language_description=str(description),
            scene_objects=scene_objects,
            metadata=metadata,
        )
        task.validate()
        return task

    @staticmethod
    def _parse_scene_objects(text: str) -> list[str]:
        match = re.search(r"Scene objects:\s*\[(.*?)\]", text)
        if not match:
            return []
        return [item.strip().strip("\"'") for item in match.group(1).split(",") if item.strip()]

    def validate(self) -> None:
        if not self.identifier:
            raise ValidationError("Task identifier cannot be empty")
        if not self.natural_language_description:
            raise ValidationError("Task description cannot be empty")
        if not isinstance(self.scene_objects, list):
            raise ValidationError("scene_objects must be a list")


@dataclass
class ActionStep:
    """A strict symbolic action step."""

    action: str
    object: str
    target: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.action or not isinstance(self.action, str):
            raise ValidationError("ActionStep.action must be a non-empty string")
        if not self.object or not isinstance(self.object, str):
            raise ValidationError("ActionStep.object must be a non-empty string")
        if self.target is not None and not isinstance(self.target, str):
            raise ValidationError("ActionStep.target must be a string when provided")


@dataclass
class ActionPlan:
    """Planner output."""

    identifier: str
    planner: str
    steps: list[ActionStep]

    def validate(self) -> None:
        if not self.identifier:
            raise ValidationError("ActionPlan.identifier cannot be empty")
        if not self.planner:
            raise ValidationError("ActionPlan.planner cannot be empty")
        if not isinstance(self.steps, list):
            raise ValidationError("ActionPlan.steps must be a list")
        for step in self.steps:
            step.validate()


@dataclass
class ExecutionEvent:
    """One execution event for traceability."""

    index: int
    action: str
    object: str
    target: str | None
    success: bool
    error: str | None = None


@dataclass
class ExecutionTrace:
    """Execution trace generated from an action plan."""

    identifier: str
    planner: str
    events: list[ExecutionEvent]
    start_time: str
    end_time: str
    final_state: dict[str, Any]

    @classmethod
    def create(cls, identifier: str, planner: str) -> ExecutionTrace:
        now = datetime.now(timezone.utc).isoformat()
        return cls(
            identifier=identifier,
            planner=planner,
            events=[],
            start_time=now,
            end_time=now,
            final_state={},
        )


@dataclass
class MetricBreakdown:
    """Metric breakdown aligned with EAI-like dimensions."""

    executability: float
    goal_satisfaction: float
    partial_success: float
    error_taxonomy: dict[str, int]


@dataclass
class EvaluationReport:
    """Evaluation report for one task run."""

    identifier: str
    success: bool
    metrics: MetricBreakdown
    summary: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "identifier": self.identifier,
            "success": self.success,
            "metrics": {
                "executability": self.metrics.executability,
                "goal_satisfaction": self.metrics.goal_satisfaction,
                "partial_success": self.metrics.partial_success,
                "error_taxonomy": self.metrics.error_taxonomy,
            },
            "summary": self.summary,
            "details": self.details,
        }
