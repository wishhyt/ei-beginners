from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_learning_journey_contains_phase_references() -> None:
    path = ROOT / "learning_journey" / "LEARNING_PATH.md"
    text = path.read_text(encoding="utf-8")

    assert "Phase 01" in text
    assert "Phase 02" in text
    assert "Phase 03" in text
    assert "Phase 04" in text
