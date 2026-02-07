"""SAC training entrypoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ei_beginners.rl.common.env_factory import EnvConfig, make_env

try:  # pragma: no cover - optional dependency
    from stable_baselines3 import SAC
except ImportError:  # pragma: no cover
    SAC = None


def train_sac(env_id: str, steps: int, output_dir: str, seed: int = 0) -> str:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    artifact = out / "sac_model.zip"

    if SAC is None:
        payload = {
            "algo": "sac",
            "env_id": env_id,
            "steps": steps,
            "seed": seed,
            "backend": "fallback",
        }
        artifact.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return str(artifact)

    env = make_env(EnvConfig(env_id=env_id, seed=seed))
    model = SAC("MlpPolicy", env, verbose=0, seed=seed)
    model.learn(total_timesteps=steps)
    model.save(str(artifact.with_suffix("")))
    env.close()
    return str(artifact)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="Pendulum-v1")
    parser.add_argument("--steps", type=int, default=1_000)
    parser.add_argument("--output", default="outputs/rl/sac")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    path = train_sac(args.env, args.steps, args.output, args.seed)
    print(path)


if __name__ == "__main__":
    main()
