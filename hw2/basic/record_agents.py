# 我们使用两个wrapper来记录agent的轨迹
# RecordEpisodeStatistics & RecordVideo

import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo

num_eval_episodes = 10

env = gym.make("CartPole-v1", render_mode="rgb_array")
env = RecordVideo(env, video_folder="cartpole-agent", name_prefix="eval", episode_trigger=lambda x:True)
env = RecordEpisodeStatistics(env, buffer_length=num_eval_episodes)

for episode_num in range(num_eval_episodes):
    obs, info = env.reset()
    episode_over = False
    while not episode_over:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        episode_over = terminated or truncated
env.close()

print(f'Episode time taken: {env.time_queue}')
print(f'Episode total rewards: {env.return_queue}')
print(f'Episode lengths: {env.length_queue}')