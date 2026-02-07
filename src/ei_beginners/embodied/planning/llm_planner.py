"""Optional LLM planner with rule fallback."""

from __future__ import annotations

import json
import os
import re

from ei_beginners.embodied.models import ActionPlan, ActionStep, TaskSpec
from ei_beginners.embodied.planning.planner_base import Planner
from ei_beginners.embodied.planning.rule_planner import RulePlanner

try:
    import google.generativeai as genai
except ImportError:  # pragma: no cover - optional dependency
    genai = None


class LLMPlanner(Planner):
    """LLM-based planner that falls back to deterministic rules."""

    def __init__(self, model_name: str = "gemini-2.5-flash") -> None:
        self.model_name = model_name
        self._fallback = RulePlanner()

    def plan(self, task: TaskSpec) -> ActionPlan:
        if genai is None or not os.getenv("GEMINI_API_KEY"):
            return self._fallback.plan(task)

        prompt = self._build_prompt(task)
        try:
            genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
            model = genai.GenerativeModel(self.model_name)
            response = model.generate_content(prompt)
            text = getattr(response, "text", "") or ""
            steps = self._parse_steps(text)
            if not steps:
                return self._fallback.plan(task)
            plan = ActionPlan(identifier=task.identifier, planner="llm", steps=steps)
            plan.validate()
            return plan
        except Exception:
            return self._fallback.plan(task)

    @staticmethod
    def _build_prompt(task: TaskSpec) -> str:
        return (
            "Create an embodied action plan in JSON array form. "
            "Each step must include action and object; target is optional. "
            f"Task: {task.natural_language_description}\n"
            f"Scene objects: {task.scene_objects}\n"
            "Output only JSON."
        )

    @staticmethod
    def _parse_steps(text: str) -> list[ActionStep]:
        block_match = re.search(r"\[[\s\S]*\]", text)
        if not block_match:
            return []
        raw = block_match.group(0)
        payload = json.loads(raw)
        steps: list[ActionStep] = []
        for item in payload:
            if not isinstance(item, dict):
                continue
            steps.append(
                ActionStep(
                    action=str(item.get("action", "")).upper(),
                    object=str(item.get("object", "")),
                    target=(
                        str(item["target"])
                        if "target" in item and item["target"] is not None
                        else None
                    ),
                    metadata={
                        k: v for k, v in item.items() if k not in {"action", "object", "target"}
                    },
                )
            )
        return steps
