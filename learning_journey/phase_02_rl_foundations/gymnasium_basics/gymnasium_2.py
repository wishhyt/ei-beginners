import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from gymnasium.wrappers import (
    TimeLimit, 
    ClipAction, 
    FlattenObservation,
    RecordEpisodeStatistics
)
from collections import defaultdict, deque
import pickle
import os
from datetime import datetime

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class GymnasiumComprehensiveDemo:
    """Gymnasium综合演示类"""
    
    def __init__(self, env_name="CartPole-v1", render_mode="human"):
        self.env_name = env_name
        self.render_mode = render_mode
        self.results_dir = "gymnasium_2_results"
        os.makedirs(self.results_dir, exist_ok=True)
        
        # 训练记录
        self.episode_rewards = []
        self.episode_lengths = []
        self.training_history = []
        
    def demonstrate_basic_usage(self):
        """演示基本的Gymnasium使用方法"""
        print("=" * 60)
        print("1. 基本环境使用演示")
        print("=" * 60)
        
        # 创建环境
        env = gym.make(self.env_name, render_mode=self.render_mode)
        
        # 环境信息
        print(f"环境名称: {self.env_name}")
        print(f"动作空间: {env.action_space}")
        print(f"观察空间: {env.observation_space}")
        print(f"动作空间类型: {type(env.action_space)}")
        print(f"观察空间类型: {type(env.observation_space)}")
        
        # 重置环境
        observation, info = env.reset(seed=42)
        print(f"初始观察: {observation}")
        print(f"初始信息: {info}")
        
        # 运行几步随机动作
        print("\n运行随机动作...")
        total_reward = 0
        
        for step in range(10):
            # 随机采样动作
            action = env.action_space.sample()
            print(f"步骤 {step}: 动作 = {action}")
            
            # 执行动作
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            print(f"  观察: {observation}")
            print(f"  奖励: {reward}")
            print(f"  终止: {terminated}, 截断: {truncated}")
            
            if terminated or truncated:
                print(f"回合结束! 总奖励: {total_reward}")
                break
        
        env.close()
        return total_reward
    
    def demonstrate_spaces(self):
        """演示不同类型的空间"""
        print("\n" + "=" * 60)
        print("2. 空间类型演示")
        print("=" * 60)
        
        # 测试不同环境的空间类型
        environments = [
            "CartPole-v1",      # Discrete action, Box observation
            "MountainCar-v0",   # Discrete action, Box observation
            "Pendulum-v1",      # Box action, Box observation
        ]
        
        for env_name in environments:
            print(f"\n--- {env_name} ---")
            try:
                env = gym.make(env_name)
                
                # 动作空间分析
                action_space = env.action_space
                print(f"动作空间: {action_space}")
                
                if hasattr(action_space, 'n'):
                    print(f"  离散动作数量: {action_space.n}")
                elif hasattr(action_space, 'shape'):
                    print(f"  连续动作形状: {action_space.shape}")
                    print(f"  动作范围: [{action_space.low}, {action_space.high}]")
                
                # 观察空间分析
                obs_space = env.observation_space
                print(f"观察空间: {obs_space}")
                print(f"  观察形状: {obs_space.shape}")
                print(f"  观察范围: [{obs_space.low}, {obs_space.high}]")
                
                # 采样测试
                action_sample = action_space.sample()
                obs_sample = obs_space.sample()
                print(f"  动作采样: {action_sample}")
                print(f"  观察采样: {obs_sample[:3]}..." if len(obs_sample) > 3 else f"  观察采样: {obs_sample}")
                
                # 验证采样的有效性
                print(f"  动作有效性: {action_space.contains(action_sample)}")
                print(f"  观察有效性: {obs_space.contains(obs_sample)}")
                
                env.close()
            except Exception as e:
                print(f"  错误: {e}")
    
    def demonstrate_wrappers(self):
        """演示包装器的使用"""
        print("\n" + "=" * 60)
        print("3. 包装器演示")
        print("=" * 60)
        
        # 创建基础环境
        base_env = gym.make(self.env_name)
        print(f"原始环境: {base_env}")
        print(f"原始观察空间: {base_env.observation_space}")
        
        # 应用多个包装器
        wrapped_env = base_env
        
        # 1. 时间限制包装器
        wrapped_env = TimeLimit(wrapped_env, max_episode_steps=500)
        print(f"添加TimeLimit后: {wrapped_env}")
        
        # 2. 动作裁剪包装器（如果是连续动作空间）
        if isinstance(base_env.action_space, gym.spaces.Box):
            wrapped_env = ClipAction(wrapped_env)
            print(f"添加ClipAction后: {wrapped_env}")
        
        # 3. 观察展平包装器（如果观察是多维的）
        if len(base_env.observation_space.shape) > 1:
            wrapped_env = FlattenObservation(wrapped_env)
            print(f"添加FlattenObservation后: {wrapped_env}")
            print(f"展平后观察空间: {wrapped_env.observation_space}")
        
        # 4. 统计记录包装器
        wrapped_env = RecordEpisodeStatistics(wrapped_env)
        print(f"添加RecordEpisodeStatistics后: {wrapped_env}")
        
        # 测试包装后的环境
        print("\n测试包装后的环境...")
        obs, info = wrapped_env.reset()
        print(f"重置后观察形状: {obs.shape if hasattr(obs, 'shape') else type(obs)}")
        
        # 运行一小段
        for i in range(5):
            action = wrapped_env.action_space.sample()
            obs, reward, terminated, truncated, info = wrapped_env.step(action)
            if terminated or truncated:
                print(f"回合在第{i+1}步结束")
                if 'episode' in info:
                    print(f"回合统计: {info['episode']}")
                break
        
        # 获取原始环境
        original_env = wrapped_env.unwrapped
        print(f"\n原始环境: {original_env}")
        
        wrapped_env.close()
        return wrapped_env
    
    def create_custom_wrapper(self):
        """创建自定义包装器"""
        
        class RewardShapingWrapper(gym.Wrapper):
            """自定义奖励塑形包装器"""
            
            def __init__(self, env, reward_scale=1.0):
                super().__init__(env)
                self.reward_scale = reward_scale
                self.step_count = 0
            
            def step(self, action):
                obs, reward, terminated, truncated, info = self.env.step(action)
                self.step_count += 1
                
                # 奖励塑形：给予存活时间奖励
                shaped_reward = reward * self.reward_scale
                if not terminated and not truncated:
                    shaped_reward += 0.01  # 存活奖励
                
                info['original_reward'] = reward
                info['shaped_reward'] = shaped_reward
                
                return obs, shaped_reward, terminated, truncated, info
            
            def reset(self, **kwargs):
                self.step_count = 0
                return self.env.reset(**kwargs)
        
        print("\n" + "=" * 60)
        print("4. 自定义包装器演示")
        print("=" * 60)
        
        # 使用自定义包装器
        env = gym.make(self.env_name)
        custom_env = RewardShapingWrapper(env, reward_scale=1.5)
        
        print("使用自定义奖励塑形包装器...")
        obs, info = custom_env.reset()
        
        total_original = 0
        total_shaped = 0
        
        for i in range(20):
            action = custom_env.action_space.sample()
            obs, reward, terminated, truncated, info = custom_env.step(action)
            
            total_original += info.get('original_reward', 0)
            total_shaped += info.get('shaped_reward', 0)
            
            if terminated or truncated:
                break
        
        print(f"原始总奖励: {total_original}")
        print(f"塑形后总奖励: {total_shaped}")
        
        custom_env.close()
        return custom_env
    
    def simple_q_learning(self, episodes=1000, render_training=False):
        """实现简单的Q-learning算法"""
        print("\n" + "=" * 60)
        print("5. Q-learning训练演示")
        print("=" * 60)
        
        # 创建离散化的环境（适用于连续观察空间）
        env = gym.make(self.env_name, render_mode="rgb_array" if not render_training else "human")
        
        # 如果是连续观察空间，需要离散化
        if isinstance(env.observation_space, gym.spaces.Box):
            # 为简单起见，我们创建一个简单的离散化方法
            obs_bins = [20] * env.observation_space.shape[0]  # 每个维度20个bin
            
            def discretize_obs(obs):
                """将连续观察离散化"""
                discrete_obs = []
                for i, (val, low, high, bins) in enumerate(zip(
                    obs, env.observation_space.low, env.observation_space.high, obs_bins)):
                    # 处理无限值
                    if np.isinf(low):
                        low = -5.0
                    if np.isinf(high):
                        high = 5.0
                    
                    # 裁剪并离散化
                    val = np.clip(val, low, high)
                    discrete_val = int((val - low) / (high - low) * (bins - 1))
                    discrete_obs.append(discrete_val)
                return tuple(discrete_obs)
        else:
            def discretize_obs(obs):
                return tuple(obs) if isinstance(obs, np.ndarray) else obs
        
        # Q-table
        q_table = defaultdict(lambda: np.zeros(env.action_space.n))
        
        # 参数
        learning_rate = 0.1
        discount_factor = 0.95
        epsilon = 1.0
        epsilon_decay = 0.995
        epsilon_min = 0.01
        
        print(f"开始训练 {episodes} 回合...")
        
        episode_rewards = []
        episode_lengths = []
        recent_rewards = deque(maxlen=100)
        
        for episode in range(episodes):
            obs, info = env.reset()
            state = discretize_obs(obs)
            
            total_reward = 0
            steps = 0
            
            while True:
                # ε-贪婪策略选择动作
                if np.random.random() < epsilon:
                    action = env.action_space.sample()  # 探索
                else:
                    action = np.argmax(q_table[state])  # 利用
                
                # 执行动作
                next_obs, reward, terminated, truncated, info = env.step(action)
                next_state = discretize_obs(next_obs)
                
                # Q-learning更新
                old_q = q_table[state][action]
                next_max_q = np.max(q_table[next_state])
                new_q = old_q + learning_rate * (reward + discount_factor * next_max_q - old_q)
                q_table[state][action] = new_q
                
                state = next_state
                total_reward += reward
                steps += 1
                
                if terminated or truncated:
                    break
            
            # 记录结果
            episode_rewards.append(total_reward)
            episode_lengths.append(steps)
            recent_rewards.append(total_reward)
            
            # 衰减探索率
            epsilon = max(epsilon_min, epsilon * epsilon_decay)
            
            # 打印进度
            if episode % 100 == 0:
                avg_reward = np.mean(recent_rewards)
                print(f"回合 {episode}, 平均奖励: {avg_reward:.2f}, ε: {epsilon:.3f}")
        
        env.close()
        
        # 保存训练结果
        self.episode_rewards = episode_rewards
        self.episode_lengths = episode_lengths
        
        # 保存Q-table
        with open(os.path.join(self.results_dir, 'q_table.pkl'), 'wb') as f:
            pickle.dump(dict(q_table), f)
        
        print(f"训练完成! Q-table大小: {len(q_table)}")
        print(f"最后100回合平均奖励: {np.mean(episode_rewards[-100:]):.2f}")
        
        return q_table
    
    def test_trained_agent(self, q_table, test_episodes=5):
        """测试训练好的智能体"""
        print("\n" + "=" * 60)
        print("6. 测试训练好的智能体")
        print("=" * 60)
        
        env = gym.make(self.env_name, render_mode=self.render_mode)
        
        # 离散化函数（与训练时相同）
        if isinstance(env.observation_space, gym.spaces.Box):
            obs_bins = [20] * env.observation_space.shape[0]
            
            def discretize_obs(obs):
                discrete_obs = []
                for i, (val, low, high, bins) in enumerate(zip(
                    obs, env.observation_space.low, env.observation_space.high, obs_bins)):
                    if np.isinf(low):
                        low = -5.0
                    if np.isinf(high):
                        high = 5.0
                    val = np.clip(val, low, high)
                    discrete_val = int((val - low) / (high - low) * (bins - 1))
                    discrete_obs.append(discrete_val)
                return tuple(discrete_obs)
        else:
            def discretize_obs(obs):
                return tuple(obs) if isinstance(obs, np.ndarray) else obs
        
        test_rewards = []
        
        for episode in range(test_episodes):
            obs, info = env.reset()
            state = discretize_obs(obs)
            
            total_reward = 0
            steps = 0
            
            print(f"\n测试回合 {episode + 1}:")
            
            while True:
                # 使用贪婪策略（不探索）
                if state in q_table:
                    action = np.argmax(q_table[state])
                else:
                    action = env.action_space.sample()  # 未见过的状态随机选择
                
                obs, reward, terminated, truncated, info = env.step(action)
                state = discretize_obs(obs)
                
                total_reward += reward
                steps += 1
                
                if terminated or truncated:
                    print(f"  回合结束: {steps} 步, 总奖励: {total_reward}")
                    break
            
            test_rewards.append(total_reward)
        
        env.close()
        
        print(f"\n测试结果:")
        print(f"平均奖励: {np.mean(test_rewards):.2f}")
        print(f"标准差: {np.std(test_rewards):.2f}")
        print(f"最大奖励: {np.max(test_rewards)}")
        print(f"最小奖励: {np.min(test_rewards)}")
        
        return test_rewards
    
    def visualize_results(self):
        """可视化训练结果"""
        print("\n" + "=" * 60)
        print("7. 结果可视化")
        print("=" * 60)
        
        if not self.episode_rewards:
            print("没有训练数据可视化")
            return
        
        # 创建图形
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'{self.env_name} 训练结果分析', fontsize=16)
        
        # 1. 回合奖励趋势
        axes[0, 0].plot(self.episode_rewards, alpha=0.6, color='blue')
        # 添加移动平均线
        window = 50
        if len(self.episode_rewards) > window:
            moving_avg = np.convolve(self.episode_rewards, 
                                   np.ones(window)/window, mode='valid')
            axes[0, 0].plot(range(window-1, len(self.episode_rewards)), 
                          moving_avg, color='red', linewidth=2, label=f'{window}回合移动平均')
        
        axes[0, 0].set_title('回合奖励趋势')
        axes[0, 0].set_xlabel('回合')
        axes[0, 0].set_ylabel('奖励')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 回合长度趋势
        axes[0, 1].plot(self.episode_lengths, alpha=0.6, color='green')
        if len(self.episode_lengths) > window:
            moving_avg_length = np.convolve(self.episode_lengths, 
                                          np.ones(window)/window, mode='valid')
            axes[0, 1].plot(range(window-1, len(self.episode_lengths)), 
                          moving_avg_length, color='red', linewidth=2, label=f'{window}回合移动平均')
        
        axes[0, 1].set_title('回合长度趋势')
        axes[0, 1].set_xlabel('回合')
        axes[0, 1].set_ylabel('步数')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 奖励分布直方图
        axes[1, 0].hist(self.episode_rewards, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[1, 0].set_title('奖励分布')
        axes[1, 0].set_xlabel('奖励')
        axes[1, 0].set_ylabel('频次')
        axes[1, 0].axvline(np.mean(self.episode_rewards), color='red', 
                         linestyle='--', label=f'平均值: {np.mean(self.episode_rewards):.2f}')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 学习曲线（最后几百回合的表现）
        last_episodes = min(200, len(self.episode_rewards))
        recent_rewards = self.episode_rewards[-last_episodes:]
        axes[1, 1].plot(range(len(self.episode_rewards)-last_episodes, len(self.episode_rewards)), 
                       recent_rewards, color='purple', alpha=0.7)
        axes[1, 1].set_title(f'最后{last_episodes}回合表现')
        axes[1, 1].set_xlabel('回合')
        axes[1, 1].set_ylabel('奖励')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图片
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.results_dir, f'training_results_{timestamp}.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"结果图片已保存到: {filename}")
        
        plt.show()
        
        # 打印统计信息
        print(f"\n统计信息:")
        print(f"总回合数: {len(self.episode_rewards)}")
        print(f"平均奖励: {np.mean(self.episode_rewards):.2f}")
        print(f"奖励标准差: {np.std(self.episode_rewards):.2f}")
        print(f"最大奖励: {np.max(self.episode_rewards)}")
        print(f"最小奖励: {np.min(self.episode_rewards)}")
        print(f"最后100回合平均奖励: {np.mean(self.episode_rewards[-100:]):.2f}")
    
    def run_comprehensive_demo(self):
        """运行完整的演示"""
        print("Gymnasium 综合强化学习任务演示")
        print("=" * 80)
        
        # 1. 基本使用
        self.demonstrate_basic_usage()
        
        # 2. 空间演示
        self.demonstrate_spaces()
        
        # 3. 包装器演示
        self.demonstrate_wrappers()
        
        # 4. 自定义包装器
        self.create_custom_wrapper()
        
        # 5. Q-learning训练
        print("\n是否开始Q-learning训练? (训练可能需要几分钟)")
        user_input = input("输入 'y' 开始训练，其他键跳过: ")
        
        if user_input.lower() == 'y':
            q_table = self.simple_q_learning(episodes=1000, render_training=False)
            
            # 6. 测试训练好的智能体
            print("\n是否测试训练好的智能体?")
            test_input = input("输入 'y' 开始测试，其他键跳过: ")
            if test_input.lower() == 'y':
                self.test_trained_agent(q_table, test_episodes=3)
            
            # 7. 可视化结果
            self.visualize_results()
        
        print("\n" + "=" * 80)
        print("演示完成! 所有结果已保存到", self.results_dir)
        print("=" * 80)

def main():
    """主函数"""
    # 创建演示实例
    # 可以尝试不同的环境：
    # - "CartPole-v1": 经典的倒立摆任务
    # - "MountainCar-v0": 爬山车任务  
    # - "LunarLander-v3": 月球登陆器任务
    
    demo = GymnasiumComprehensiveDemo(
        env_name="CartPole-v1",
        render_mode="human"  # 设置为 "rgb_array" 来关闭可视化
    )
    
    # 运行完整演示
    demo.run_comprehensive_demo()

if __name__ == "__main__":
    main() 