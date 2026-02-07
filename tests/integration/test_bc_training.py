from __future__ import annotations

import json
from pathlib import Path

from ei_beginners.rl.bc.train import train_bc

ROOT = Path(__file__).resolve().parents[2]


def test_bc_trainer_loads_trajectory_and_runs_epoch(tmp_path: Path) -> None:
    dataset = ROOT / "learning_journey" / "phase_04_eai_evaluation" / "samples" / "ans1.json"
    artifact = Path(train_bc(dataset=str(dataset), output_dir=str(tmp_path), epochs=1))
    assert artifact.exists()

    payload = json.loads(artifact.read_text(encoding="utf-8"))
    assert payload["algorithm"] == "behavior_cloning"
    assert payload["epochs"] == 1
    assert payload["samples"] >= 1
