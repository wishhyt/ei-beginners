from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_quality_gate_tool_configs_exist() -> None:
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "[tool.ruff]" in text
    assert "[tool.black]" in text
    assert "[tool.mypy]" in text
    assert "pytest" in text
