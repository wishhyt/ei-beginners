from __future__ import annotations

from pathlib import Path

from ei_beginners.rl.dqn.train import train_dqn
from ei_beginners.rl.ppo.train import train_ppo
from ei_beginners.rl.sac.train import train_sac


def test_rl_trainers_create_artifacts(tmp_path: Path) -> None:
    dqn_artifact = Path(
        train_dqn("CartPole-v1", steps=10, output_dir=str(tmp_path / "dqn"), seed=0)
    )
    ppo_artifact = Path(
        train_ppo("CartPole-v1", steps=10, output_dir=str(tmp_path / "ppo"), seed=0)
    )
    sac_artifact = Path(
        train_sac("Pendulum-v1", steps=10, output_dir=str(tmp_path / "sac"), seed=0)
    )

    assert dqn_artifact.exists()
    assert ppo_artifact.exists()
    assert sac_artifact.exists()
