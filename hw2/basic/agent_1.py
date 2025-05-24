# epsilon-greedy 策略
# 如果生成的随机数小于epsilon，则执行探索；若大于epsilon，则执行利用，基于Q-table，选择Q最大值对应的action

from collections import defaultdict
import gymnasium as gym
import numpy as np
import os
import pickle
from datetime import datetime

class BlackjackAgent:
    def __init__(
        self,
        env: gym.Env,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.95,
    ):
        """Initialize a Reinforcement Learning agent with an empty dictionary
        of state-action values (q_values), a learning rate and an epsilon.

        Args:
            env: The training environment
            learning_rate: The learning rate
            initial_epsilon: The initial epsilon value
            epsilon_decay: The decay for epsilon
            final_epsilon: The final epsilon value
            discount_factor: The discount factor for computing the Q-value
        """
        self.env = env
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))

        self.lr = learning_rate
        self.discount_factor = discount_factor

        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        self.training_error = []
    # 这里是注解语法，表示参数obs的类型为tuple[int,int,bool]，函数返回值为int
    def get_action(self,obs:tuple[int,int,bool]) -> int:
        """
        Returns the best action with probability (1 - epsilon)
        otherwise a random action with probability epsilon to ensure exploration.
        """
        # with probability epsilon return a random action to explore the environment
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        # with probability (1 - epsilon) act greedily (exploit)
        else:
            return int(np.argmax(self.q_values[obs]))

    def update(
        self,
        obs: tuple[int, int, bool],
        action: int,
        reward: float,
        terminated: bool,
        next_obs: tuple[int, int, bool],
    ):
        """Updates the Q-value of an action."""
        future_q_value = (not terminated) * np.max(self.q_values[next_obs])
        temporal_difference = (
            reward + self.discount_factor * future_q_value - self.q_values[obs][action]
        )

        self.q_values[obs][action] = (
            self.q_values[obs][action] + self.lr * temporal_difference
        )
        self.training_error.append(temporal_difference)

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)

# after each game round, Q will be updated
# hyperparameters
learning_rate = 0.01
n_episodes = 100_000
start_epsilon = 1.0
epsilon_decay = start_epsilon / (n_episodes / 2)  # reduce the exploration over time
final_epsilon = 0.1

env = gym.make("Blackjack-v1", sab=False)
env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)

agent = BlackjackAgent(
    env=env,
    learning_rate=learning_rate,
    initial_epsilon=start_epsilon,
    epsilon_decay=epsilon_decay,
    final_epsilon=final_epsilon,
)

""" 开始训练
    ↓
for 每个回合:
    ├─ 重置环境
    ├─ while 回合未结束:
    │   ├─ 智能体选择动作
    │   ├─ 环境执行动作
    │   ├─ 智能体学习更新
    │   └─ 更新状态
    └─ 衰减探索率
    ↓
训练结束 """

from tqdm import tqdm

for episode in tqdm(range(n_episodes)):
    obs, info = env.reset()
    done = False

    # play one episode
    while not done:
        action = agent.get_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)

        # update the agent
        agent.update(obs, action, reward, terminated, next_obs)

        # update if the environment is done and the current obs
        done = terminated or truncated
        obs = next_obs

    agent.decay_epsilon()

# 创建结果保存目录
results_dir = "agent_results"
os.makedirs(results_dir, exist_ok=True)

# 保存训练好的 Q-table
with open(os.path.join(results_dir, 'blackjack_q_table.pkl'), 'wb') as f:
    pickle.dump(dict(agent.q_values), f)

print(f"Q-table 已保存到 {results_dir}/blackjack_q_table.pkl")
print(f"Q-table 大小: {len(agent.q_values)} 个状态")

from matplotlib import pyplot as plt

def get_moving_avgs(arr, window, convolution_mode):
    return np.convolve(
        np.array(arr).flatten(),
        np.ones(window),
        mode=convolution_mode
    ) / window

# Smooth over a 500 episode window
rolling_length = 500
fig, axs = plt.subplots(ncols=3, figsize=(12, 5))

axs[0].set_title("Episode rewards")
reward_moving_average = get_moving_avgs(
    env.return_queue,
    rolling_length,
    "valid"
)
axs[0].plot(range(len(reward_moving_average)), reward_moving_average)

axs[1].set_title("Episode lengths")
length_moving_average = get_moving_avgs(
    env.length_queue,
    rolling_length,
    "valid"
)
axs[1].plot(range(len(length_moving_average)), length_moving_average)

axs[2].set_title("Training Error")
training_error_moving_average = get_moving_avgs(
    agent.training_error,
    rolling_length,
    "same"
)
axs[2].plot(range(len(training_error_moving_average)), training_error_moving_average)
plt.tight_layout()

# 保存图表
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
figure_filename = os.path.join(results_dir, f'blackjack_training_results_{timestamp}.png')
plt.savefig(figure_filename, dpi=300, bbox_inches='tight')
print(f"训练结果图表已保存到: {figure_filename}")

plt.show()

# 保存训练统计数据
training_stats = {
    'n_episodes': n_episodes,
    'learning_rate': learning_rate,
    'start_epsilon': start_epsilon,
    'final_epsilon': final_epsilon,
    'episode_rewards': list(env.return_queue),
    'episode_lengths': list(env.length_queue),
    'training_errors': agent.training_error,
    'final_q_table_size': len(agent.q_values)
}

stats_filename = os.path.join(results_dir, f'training_stats_{timestamp}.pkl')
with open(stats_filename, 'wb') as f:
    pickle.dump(training_stats, f)

print(f"训练统计数据已保存到: {stats_filename}")

# 打印最终统计信息
print(f"\n=== 训练完成统计 ===")
print(f"总回合数: {n_episodes}")
print(f"Q-table 大小: {len(agent.q_values)} 个状态")
print(f"平均回合奖励: {np.mean(env.return_queue):.4f}")
print(f"最后1000回合平均奖励: {np.mean(list(env.return_queue)[-1000:]):.4f}")
print(f"平均回合长度: {np.mean(env.length_queue):.2f}")
print(f"所有结果已保存到: {results_dir}")
print("=" * 25)