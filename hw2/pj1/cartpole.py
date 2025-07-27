import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

# 改进超参数
learning_rate = 0.1  # α，保持稳定
discount_factor = 0.99  # γ
epsilon = 1.0  # 初始探索率
epsilon_min = 0.01
epsilon_decay = 0.999  # 缓和衰减，便于长期探索
episodes = 10000  # 增加以确保收敛
test_episodes = 100  # 测试episode数

# 状态离散化参数改进
buckets = (20, 20, 40, 40)  # 增加分辨率，平衡大小与精度

# 初始化环境
env = gym.make("CartPole-v1")
n_actions = env.action_space.n  # 动作空间：2

# Q表
q_table = np.zeros(buckets + (n_actions,))

# 改进状态离散化函数：边界扩展
def discretize_state(state):
    upper_bounds = [2.5, 10, 0.5, 10]  # 覆盖实际范围
    lower_bounds = [-2.5, -10, -0.5, -10]
    ratios = [(state[i] + abs(lower_bounds[i])) / (upper_bounds[i] - lower_bounds[i]) for i in range(len(state))]
    new_state = [int(round((buckets[i] - 1) * ratios[i])) for i in range(len(state))]
    new_state = [min(buckets[i] - 1, max(0, new_state[i])) for i in range(len(state))]
    return tuple(new_state)

# 训练过程
rewards = []
for episode in range(episodes):
    state, _ = env.reset()
    state = discretize_state(state)
    total_reward = 0
    done = False
    while not done:
        # ε-greedy选择动作
        if np.random.rand() < epsilon:
            action = env.action_space.sample()  # 探索
        else:
            action = np.argmax(q_table[state])  # 利用
        
        next_state, reward, done, _, _ = env.step(action)
        next_state = discretize_state(next_state)
        total_reward += reward
        
        # Q值更新
        best_next_action = np.argmax(q_table[next_state])
        q_table[state][action] += learning_rate * (reward + discount_factor * q_table[next_state][best_next_action] - q_table[state][action])
        
        state = next_state
    
    # 衰减ε
    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    rewards.append(total_reward)
    if (episode + 1) % 100 == 0:
        print(f"Episode {episode + 1}: Average reward over last 100: {np.mean(rewards[-100:])}")

# 绘制并保存学习曲线图片（字体默认英文）
plt.plot(rewards)
plt.title("Training Rewards over Episodes")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.savefig("cartpole_training_reward.png")  # 保存到同目录
plt.close()  # 关闭图形，避免显示

# 测试过程
success_count = 0
test_rewards = []
for _ in range(test_episodes):
    state, _ = env.reset()
    state = discretize_state(state)
    total_reward = 0
    done = False
    while not done:
        action = np.argmax(q_table[state])
        next_state, reward, done, _, _ = env.step(action)
        next_state = discretize_state(next_state)
        total_reward += reward
        state = next_state
    test_rewards.append(total_reward)
    if total_reward >= 195:  # 接近最大长度视为成功
        success_count += 1

print(f"Test Average Reward: {np.mean(test_rewards)}")
print(f"Success Rate: {success_count / test_episodes * 100}%")

# 可选：渲染一个测试episode（可视化）
env = gym.make("CartPole-v1", render_mode="human")
state, _ = env.reset()
state = discretize_state(state)
done = False
while not done:
    action = np.argmax(q_table[state])
    next_state, _, done, _, _ = env.step(action)
    next_state = discretize_state(next_state)
    state = next_state
    env.render()
env.close()