"""Environment factory for RL experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class EnvConfig:
    env_id: str = "CartPole-v1"
    render_mode: str | None = None
    seed: int = 0


def make_env(config: EnvConfig) -> Any:
    try:
        import gymnasium as gym
    except ImportError as exc:  # pragma: no cover - dependency optional at runtime
        raise RuntimeError("gymnasium is required for RL training") from exc

    env = gym.make(config.env_id, render_mode=config.render_mode)
    env.reset(seed=config.seed)
    return env
