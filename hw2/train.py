import os
import argparse
import numpy as np

import gymnasium as gym
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from envs.panda_grasp_env import make_env


def make_single_env(render_mode=None, seed=0, **env_kwargs):
    def _init():
        env = make_env(render_mode=render_mode, **env_kwargs)
        env = Monitor(env)
        env.reset(seed=seed)
        return env
    return _init


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, default="ppo", choices=["ppo", "sac"])
    parser.add_argument("--total-steps", type=int, default=2_000_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--logdir", type=str, default="runs/exp1")
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--eval-freq", type=int, default=25_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-dr", action="store_true", help="关闭域随机化")
    parser.add_argument("--no-normalize", action="store_true", help="关闭观测/回报归一化(VecNormalize)")
    parser.add_argument("--use-sde", action="store_true", help="对连续动作使用 State-Dependent Exploration(PPO)")
    parser.add_argument("--env-backend", type=str, default="dummy", choices=["dummy", "pybullet"], help="环境后端")
    args = parser.parse_args()

    os.makedirs(args.logdir, exist_ok=True)

    # 并行环境（子进程，适合 PyBullet DIRECT）
    base_vec_env = SubprocVecEnv([
        make_single_env(
            render_mode=None,
            seed=args.seed + i,
            domain_randomization=not args.no_dr,
            max_steps=150,
            backend=args.env_backend,
        ) for i in range(args.n_envs)
    ])

    # 观察/奖励归一化（强烈建议开启以提升稳定性与成功率）
    if args.no_normalize:
        venv = base_vec_env
        eval_vec_env = DummyVecEnv([make_single_env(render_mode=None, seed=args.seed + 10_000, domain_randomization=False, max_steps=150)])
    else:
        venv = VecNormalize(base_vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)
        # 为评估构建一个单环境 VecNormalize，并共享 obs_rms（评估阶段不更新统计量）
        eval_vec_env = DummyVecEnv([make_single_env(render_mode=None, seed=args.seed + 10_000, domain_randomization=False, max_steps=150, backend=args.env_backend)])
        eval_vec_env = VecNormalize(eval_vec_env, training=False, norm_reward=False, norm_obs=True, clip_obs=10.0)
        eval_vec_env.obs_rms = venv.obs_rms

    # 评估环境（单个，用于回调）
    # EvalCallback 期望传入 gym.Env 或 VecEnv。这里传入与训练统计共享的 VecEnv
    eval_env = eval_vec_env

    # 算法选择与常用超参数
    if args.algo == "ppo":
        model = PPO(
            "MlpPolicy", venv,
            n_steps=1024, batch_size=1024, gae_lambda=0.95, gamma=0.99,
            learning_rate=3e-4, clip_range=0.2, ent_coef=0.0,
            use_sde=args.use_sde, sde_sample_freq=4,
            verbose=1, tensorboard_log=args.logdir, seed=args.seed
        )
    else:  # SAC
        model = SAC(
            "MlpPolicy", venv,
            buffer_size=500_000, batch_size=256, gamma=0.99, tau=0.005,
            learning_rate=3e-4, train_freq=1, gradient_steps=1,
            learning_starts=10_000,
            verbose=1, tensorboard_log=args.logdir, seed=args.seed
        )

    # 回调：定期评估与保存
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(args.logdir, "best"),
        log_path=os.path.join(args.logdir, "eval"),
        eval_freq=max(args.eval_freq // args.n_envs, 1),
        deterministic=True,
        render=False,
        n_eval_episodes=args.eval_episodes,
    )
    ckpt_cb = CheckpointCallback(save_freq=max(args.eval_freq // args.n_envs, 1),
                                 save_path=os.path.join(args.logdir, "ckpt"),
                                 name_prefix="model")

    model.learn(total_timesteps=args.total_steps, callback=[eval_cb, ckpt_cb])

    model.save(os.path.join(args.logdir, f"{args.algo}_final"))
    # 保存 VecNormalize 统计量，便于推理/评估复现训练分布
    if isinstance(venv, VecNormalize):
        venv.save(os.path.join(args.logdir, "vecnormalize.pkl"))
    venv.close()
    eval_env.close()


if __name__ == "__main__":
    main()
