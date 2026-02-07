"""Metric computation for embodied plan execution."""

from __future__ import annotations

import re

from ei_beginners.embodied.models import EvaluationReport, ExecutionTrace, MetricBreakdown, TaskSpec


def _extract_expected_goal(task: TaskSpec) -> tuple[list[str], str | None]:
    """Extract expected moved objects and target container from NL text."""
    text = task.natural_language_description.lower()

    container = None
    for obj in task.scene_objects:
        if obj.lower().startswith(("basket", "tray", "bowl")):
            if obj.lower() in text or container is None:
                container = obj

    referenced = [obj for obj in task.scene_objects if obj.lower() in text and obj != container]
    if referenced:
        return sorted(set(referenced)), container

    # fallback from names in llm prompt style
    matched_names = re.findall(r"\b[a-z]+_\d+\b", text)
    candidates = [obj for obj in task.scene_objects if obj in matched_names and obj != container]
    return sorted(set(candidates)), container


def evaluate(task: TaskSpec, trace: ExecutionTrace) -> EvaluationReport:
    """Evaluate trace with EAI-style metrics."""
    total = len(trace.events)
    success_count = sum(1 for event in trace.events if event.success)
    executability = success_count / total if total else 0.0

    expected_objects, expected_container = _extract_expected_goal(task)
    if expected_container is not None:
        placed = trace.final_state.get("container_contents", {}).get(expected_container, [])
    else:
        placed = []

    if expected_objects:
        satisfied = sum(1 for obj in expected_objects if obj in placed)
        goal_satisfaction = satisfied / len(expected_objects)
    else:
        # if no goal parsing available, degrade to executability
        goal_satisfaction = executability

    partial_success = 0.5 * executability + 0.5 * goal_satisfaction

    taxonomy: dict[str, int] = {}
    for event in trace.events:
        if event.error:
            taxonomy[event.error] = taxonomy.get(event.error, 0) + 1

    metrics = MetricBreakdown(
        executability=round(executability, 4),
        goal_satisfaction=round(goal_satisfaction, 4),
        partial_success=round(partial_success, 4),
        error_taxonomy=taxonomy,
    )

    success = metrics.goal_satisfaction >= 1.0 and metrics.executability >= 1.0
    summary = (
        f"Executability={metrics.executability:.2f}, "
        f"GoalSatisfaction={metrics.goal_satisfaction:.2f}, "
        f"PartialSuccess={metrics.partial_success:.2f}"
    )

    return EvaluationReport(
        identifier=task.identifier,
        success=success,
        metrics=metrics,
        summary=summary,
        details={
            "expected_objects": expected_objects,
            "expected_container": expected_container,
            "placed_objects": placed,
            "event_count": total,
        },
    )
