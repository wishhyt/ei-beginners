import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 导入自定义环境以确保注册
from env_1 import GridWorldEnv

def test_basic_functionality():
    """测试基本功能"""
    print("=" * 60)
    print("1. 基本环境功能测试")
    print("=" * 60)
    
    try:
        env = gym.make("gymnasium_env/Gridworld-v114514")
        print("✓ 环境创建成功")
        
        # 显示环境信息
        print(f"动作空间: {env.action_space}")
        print(f"观察空间: {env.observation_space}")
        
        observation, info = env.reset()
        print(f"✓ 环境重置成功")
        print(f"初始观察: {observation}")
        print(f"初始信息: {info}")
        
        env.close()
        return True
        
    except Exception as e:
        print(f"✗ 环境测试失败: {e}")
        return False

def test_random_episodes():
    """测试随机回合"""
    print("\n" + "=" * 60)
    print("2. 随机智能体多回合测试")
    print("=" * 60)
    
    env = gym.make("gymnasium_env/Gridworld-v114514")
    
    episode_rewards = []
    episode_lengths = []
    
    num_episodes = 10
    
    for episode in range(num_episodes):
        observation, info = env.reset()
        total_reward = 0
        steps = 0
        
        print(f"\n回合 {episode + 1}:")
        print(f"  智能体位置: {observation['agent']}")
        print(f"  目标位置: {observation['target']}")
        
        while steps < 100:  # 最大步数限制
            action = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            
            if terminated or truncated:
                print(f"  ✓ 成功到达目标! 步数: {steps}, 奖励: {total_reward}")
                break
        
        if steps >= 100:
            print(f"  ✗ 超时结束，步数: {steps}")
            
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
    
    print(f"\n=== 统计结果 ===")
    print(f"总回合数: {num_episodes}")
    print(f"成功次数: {sum(episode_rewards)}")
    print(f"成功率: {np.mean(episode_rewards) * 100:.1f}%")
    print(f"平均步数: {np.mean(episode_lengths):.2f}")
    print(f"最少步数: {min(episode_lengths)}")
    print(f"最多步数: {max(episode_lengths)}")
    
    env.close()
    return episode_rewards, episode_lengths

def test_single_episode_detailed():
    """详细的单回合测试"""
    print("\n" + "=" * 60)
    print("3. 详细单回合测试")
    print("=" * 60)
    
    env = gym.make("gymnasium_env/Gridworld-v114514")
    
    observation, info = env.reset(seed=42)
    # 从观察空间推断环境大小
    env_size = env.observation_space['agent'].high[0] + 1
    print(f"环境大小: {env_size}x{env_size}")
    print(f"起始智能体位置: {observation['agent']}")
    print(f"目标位置: {observation['target']}")
    print(f"起始距离: {info['distance']}")
    
    action_names = {0: "右", 1: "上", 2: "左", 3: "下"}
    
    step_count = 0
    print(f"\n步骤详情:")
    
    while step_count < 20:  # 最多显示20步
        action = env.action_space.sample()
        old_pos = observation['agent'].copy()
        
        observation, reward, terminated, truncated, info = env.step(action)
        step_count += 1
        
        print(f"步骤 {step_count}: 动作={action}({action_names[action]}) | "
              f"位置 {old_pos} -> {observation['agent']} | "
              f"奖励={reward} | 距离={info['distance']:.1f}")
        
        if terminated or truncated:
            print(f"回合结束! 智能体到达目标!")
            break
    
    env.close()

def visualize_grid_world():
    """可视化网格世界"""
    print("\n" + "=" * 60)
    print("4. 环境可视化")
    print("=" * 60)
    
    try:
        env = gym.make("gymnasium_env/Gridworld-v114514")
        observation, info = env.reset(seed=42)
        
        # 从观察空间推断环境大小
        env_size = env.observation_space['agent'].high[0] + 1
        
        # 创建可视化
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # 创建网格
        grid = np.zeros((env_size, env_size))
        
        # 标记位置
        agent_pos = observation['agent']
        target_pos = observation['target']
        
        grid[agent_pos[1], agent_pos[0]] = 1  # 智能体
        grid[target_pos[1], target_pos[0]] = 2  # 目标
        
        # 绘制
        im = ax.imshow(grid, cmap='RdYlBu', alpha=0.8)
        
        # 添加网格线
        ax.set_xticks(np.arange(-0.5, env_size, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, env_size, 1), minor=True)
        ax.grid(which="minor", color="black", linestyle='-', linewidth=2)
        
        # 添加标签
        ax.text(agent_pos[0], agent_pos[1], 'A', ha='center', va='center', 
                fontsize=20, fontweight='bold', color='white')
        ax.text(target_pos[0], target_pos[1], 'T', ha='center', va='center', 
                fontsize=20, fontweight='bold', color='white')
        
        # 设置标题
        ax.set_title(f'GridWorld 环境可视化 ({env_size}x{env_size})', fontsize=16)
        ax.set_xlabel('X坐标')
        ax.set_ylabel('Y坐标')
        
        # 添加图例
        legend_elements = [
            plt.Rectangle((0,0),1,1, facecolor='blue', alpha=0.8, label='智能体 (A)'),
            plt.Rectangle((0,0),1,1, facecolor='red', alpha=0.8, label='目标 (T)'),
            plt.Rectangle((0,0),1,1, facecolor='yellow', alpha=0.8, label='空白区域')
        ]
        ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3)
        
        plt.tight_layout()
        
        # 保存图片
        os.makedirs("env_test_results", exist_ok=True)
        filename = "env_test_results/gridworld_visualization.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"✓ 可视化图片已保存到: {filename}")
        
        plt.show()
        
        env.close()
        return True
        
    except Exception as e:
        print(f"✗ 可视化失败: {e}")
        return False

def run_original_test():
    """运行您的原始测试代码"""
    print("\n" + "=" * 60)
    print("5. 原始随机测试 (1000步)")
    print("=" * 60)
    
    env = gym.make("gymnasium_env/Gridworld-v114514")
    
    observation, info = env.reset()
    total_steps = 0
    total_rewards = 0
    episodes_completed = 0
    
    print("开始1000步随机测试...")
    
    for step in range(1000):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        total_steps += 1
        total_rewards += reward
        
        if terminated or truncated:
            episodes_completed += 1
            if episodes_completed <= 5:  # 只显示前5个完成的回合
                print(f"  回合 {episodes_completed} 完成于步骤 {step + 1}")
            observation, info = env.reset()
    
    print(f"\n=== 1000步测试结果 ===")
    print(f"总步数: {total_steps}")
    print(f"完成回合数: {episodes_completed}")
    print(f"总奖励: {total_rewards}")
    print(f"平均每回合步数: {total_steps / max(episodes_completed, 1):.2f}")
    
    env.close()

def main():
    """主测试函数"""
    print("GridWorld 环境综合测试")
    print("=" * 80)
    
    # 运行所有测试
    test_results = []
    
    # 1. 基本功能测试
    test_results.append(test_basic_functionality())
    
    # 2. 多回合测试
    episode_rewards, episode_lengths = test_random_episodes()
    test_results.append(len(episode_rewards) > 0)
    
    # 3. 详细单回合测试
    test_single_episode_detailed()
    
    # 4. 可视化测试
    test_results.append(visualize_grid_world())
    
    # 5. 原始测试
    run_original_test()
    
    # 总结
    print("\n" + "=" * 80)
    print("测试总结")
    print("=" * 80)
    
    test_names = ["基本功能", "多回合测试", "环境可视化"]
    for i, (name, result) in enumerate(zip(test_names, test_results)):
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{i+1}. {name}: {status}")
    
    success_rate = sum(test_results) / len(test_results) * 100
    print(f"\n总体成功率: {success_rate:.1f}%")
    
    if all(test_results):
        print("所有测试通过! GridWorld 环境运行正常!")
    else:
        print(" 部分测试失败，请检查环境实现")
    
    print(f"\n测试结果已保存到 env_test_results 文件夹")

if __name__ == "__main__":
    main()
