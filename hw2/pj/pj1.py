"""
强化学习项目：带注意力机制的多任务导航智能体
项目结构：
1. 自定义环境：动态多目标导航环境
2. 现代算法：带注意力机制的PPO
3. 高级功能：多任务学习、课程学习、经验回放
4. 完整流程：训练、评估、可视化
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import gymnasium as gym
from gymnasium import spaces
from collections import deque, namedtuple
import random
import math
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import pickle
import os

# 设置随机种子以确保可复现性
def set_seed(seed: int = 42):
    """设置全局随机种子"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

set_seed(42)

# 1. 数据结构定义

@dataclass
class Config:
    """配置参数集合"""
    # 环境参数
    grid_size: int = 10
    max_obstacles: int = 8
    max_targets: int = 4
    max_steps: int = 200
    
    # 网络参数
    hidden_dim: int = 256
    attention_dim: int = 128
    num_heads: int = 4
    
    # 训练参数
    lr: float = 3e-4
    gamma: float = 0.99
    eps_clip: float = 0.2
    update_epochs: int = 10
    batch_size: int = 64
    
    # 课程学习参数
    curriculum_steps: int = 50000
    easy_prob: float = 0.8
    
    # 其他
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    save_interval: int = 1000

config = Config()

Experience = namedtuple('Experience', 
                       ['state', 'action', 'reward', 'next_state', 'done', 'log_prob', 'value'])

# 2. 自定义环境：动态多目标导航


class DynamicNavigationEnv(gym.Env):
    """
    动态多目标导航环境
    
    特点：
    - 智能体需要按序访问多个目标
    - 动态移动的障碍物
    - 不同难度级别
    - 丰富的状态表示
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}
    
    def __init__(self, config: Config, render_mode: Optional[str] = None):
        super().__init__()
        self.config = config
        self.size = config.grid_size
        self.max_obstacles = config.max_obstacles
        self.max_targets = config.max_targets
        self.max_steps = config.max_steps
        self.render_mode = render_mode
        
        # 动作空间：上下左右 + 等待
        self.action_space = spaces.Discrete(5)
        
        # 观察空间：agent位置 + 目标列表 + 障碍物列表 + 任务信息
        self.observation_space = spaces.Dict({
            'agent_pos': spaces.Box(0, self.size-1, shape=(2,), dtype=np.float32),
            'targets': spaces.Box(-1, self.size-1, shape=(self.max_targets, 3), dtype=np.float32),  # [x, y, visited]
            'obstacles': spaces.Box(-1, self.size-1, shape=(self.max_obstacles, 4), dtype=np.float32),  # [x, y, vx, vy]
            'task_info': spaces.Box(0, 1, shape=(4,), dtype=np.float32),  # [progress, difficulty, time_ratio, success_rate]
        })
        
        # 动作映射
        self._action_to_direction = {
            0: np.array([0, 1]),   # 上
            1: np.array([0, -1]),  # 下  
            2: np.array([-1, 0]),  # 左
            3: np.array([1, 0]),   # 右
            4: np.array([0, 0]),   # 等待
        }
        
        # 环境状态
        self.reset()
        
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        """重置环境"""
        super().reset(seed=seed)
        
        # 解析选项
        difficulty = options.get('difficulty', 0.5) if options else 0.5
        
        # 初始化智能体位置
        self.agent_pos = np.array([0, 0], dtype=np.float32)
        
        # 生成目标点（按序访问）
        self.targets = []
        num_targets = max(2, int(self.max_targets * difficulty))
        for i in range(num_targets):
            while True:
                pos = self.np_random.uniform(2, self.size-2, 2)
                # 确保目标点不重叠
                if all(np.linalg.norm(pos - t['pos']) > 1.5 for t in self.targets):
                    self.targets.append({
                        'pos': pos,
                        'visited': False,
                        'order': i
                    })
                    break
        
        # 生成动态障碍物
        self.obstacles = []
        num_obstacles = int(self.max_obstacles * difficulty)
        for i in range(num_obstacles):
            pos = self.np_random.uniform(1, self.size-1, 2)
            velocity = self.np_random.uniform(-0.5, 0.5, 2)
            self.obstacles.append({
                'pos': pos,
                'vel': velocity,
                'radius': 0.3
            })
        
        # 任务状态
        self.current_target_idx = 0
        self.steps = 0
        self.total_reward = 0
        self.difficulty = difficulty
        self.success_count = 0
        
        return self._get_obs(), self._get_info()
    
    def step(self, action: int):
        """执行动作"""
        self.steps += 1
        
        # 移动智能体
        direction = self._action_to_direction[action]
        new_pos = self.agent_pos + direction * 0.2
        
        # 边界检查
        new_pos = np.clip(new_pos, 0, self.size-1)
        
        # 碰撞检测
        collision = False
        for obs in self.obstacles:
            if np.linalg.norm(new_pos - obs['pos']) < obs['radius'] + 0.2:
                collision = True
                break
        
        if not collision:
            self.agent_pos = new_pos
        
        # 更新动态障碍物
        self._update_obstacles()
        
        # 检查目标访问
        reward = self._check_targets()
        
        # 时间惩罚
        reward -= 0.01
        
        # 碰撞惩罚
        if collision:
            reward -= 0.5
        
        # 检查任务完成
        done = (self.current_target_idx >= len(self.targets)) or (self.steps >= self.max_steps)
        
        if done and self.current_target_idx >= len(self.targets):
            reward += 10.0  # 完成奖励
            self.success_count += 1
        
        self.total_reward += reward
        
        return self._get_obs(), reward, done, False, self._get_info()
    
    def _update_obstacles(self):
        """更新动态障碍物位置"""
        for obs in self.obstacles:
            # 更新位置
            obs['pos'] += obs['vel']
            
            # 边界反弹
            for i in range(2):
                if obs['pos'][i] <= 0 or obs['pos'][i] >= self.size:
                    obs['vel'][i] *= -1
                    obs['pos'][i] = np.clip(obs['pos'][i], 0, self.size)
    
    def _check_targets(self) -> float:
        """检查目标访问"""
        if self.current_target_idx >= len(self.targets):
            return 0.0
        
        current_target = self.targets[self.current_target_idx]
        distance = np.linalg.norm(self.agent_pos - current_target['pos'])
        
        if distance < 0.5:  # 到达目标
            current_target['visited'] = True
            self.current_target_idx += 1
            return 5.0  # 目标奖励
        
        # 距离奖励塑形
        max_dist = np.sqrt(2) * self.size
        return 0.1 * (1 - distance / max_dist)
    
    def _get_obs(self) -> Dict:
        """获取观察"""
        # 目标信息（填充到固定长度）
        targets_array = np.full((self.max_targets, 3), -1, dtype=np.float32)
        for i, target in enumerate(self.targets):
            targets_array[i] = [target['pos'][0], target['pos'][1], float(target['visited'])]
        
        # 障碍物信息（填充到固定长度）
        obstacles_array = np.full((self.max_obstacles, 4), -1, dtype=np.float32)
        for i, obs in enumerate(self.obstacles):
            obstacles_array[i] = [obs['pos'][0], obs['pos'][1], obs['vel'][0], obs['vel'][1]]
        
        # 任务信息
        progress = self.current_target_idx / len(self.targets) if self.targets else 1.0
        time_ratio = self.steps / self.max_steps
        success_rate = self.success_count / max(1, self.steps // self.max_steps + 1)
        
        task_info = np.array([progress, self.difficulty, time_ratio, success_rate], dtype=np.float32)
        
        return {
            'agent_pos': self.agent_pos.copy(),
            'targets': targets_array,
            'obstacles': obstacles_array,
            'task_info': task_info
        }
    
    def _get_info(self) -> Dict:
        """获取额外信息"""
        return {
            'current_target_idx': self.current_target_idx,
            'total_targets': len(self.targets),
            'steps': self.steps,
            'difficulty': self.difficulty,
            'success_rate': self.success_count / max(1, self.steps // self.max_steps + 1)
        }
    
    def render(self):
        """渲染环境（简化版）"""
        if self.render_mode == "human":
            fig, ax = plt.subplots(figsize=(8, 8))
            
            # 绘制网格
            ax.set_xlim(0, self.size)
            ax.set_ylim(0, self.size)
            ax.set_aspect('equal')
            
            # 绘制智能体
            agent_circle = patches.Circle(self.agent_pos, 0.2, color='blue', alpha=0.8)
            ax.add_patch(agent_circle)
            
            # 绘制目标
            for i, target in enumerate(self.targets):
                color = 'green' if target['visited'] else 'red'
                alpha = 0.5 if target['visited'] else 1.0
                target_circle = patches.Circle(target['pos'], 0.3, color=color, alpha=alpha)
                ax.add_patch(target_circle)
                ax.text(target['pos'][0], target['pos'][1], str(i), ha='center', va='center')
            
            # 绘制障碍物
            for obs in self.obstacles:
                obs_circle = patches.Circle(obs['pos'], obs['radius'], color='black', alpha=0.6)
                ax.add_patch(obs_circle)
            
            plt.title(f"步骤: {self.steps}, 目标: {self.current_target_idx}/{len(self.targets)}")
            plt.show()

# 3. 注意力机制

class MultiHeadAttention(nn.Module):
    """多头注意力机制"""
    
    def __init__(self, input_dim: int, attention_dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.attention_dim = attention_dim
        self.head_dim = attention_dim // num_heads
        
        assert attention_dim % num_heads == 0
        
        self.query = nn.Linear(input_dim, attention_dim)
        self.key = nn.Linear(input_dim, attention_dim)
        self.value = nn.Linear(input_dim, attention_dim)
        self.output = nn.Linear(attention_dim, attention_dim)
        
        self.norm = nn.LayerNorm(attention_dim)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        Args:
            x: [batch_size, seq_len, input_dim]
            mask: [batch_size, seq_len] 可选的掩码
        """
        batch_size, seq_len, _ = x.shape
        
        # 计算 Q, K, V
        Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # 应用掩码
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1)  # [batch_size, 1, 1, seq_len]
            scores.masked_fill_(mask == 0, -1e9)
        
        # Softmax
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 应用注意力
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.attention_dim)
        
        # 输出投影和残差连接
        output = self.output(context)
        return self.norm(output + x[:, :, :self.attention_dim] if x.shape[-1] >= self.attention_dim else output)

class AttentionPolicyNetwork(nn.Module):
    """带注意力机制的策略网络"""
    
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        
        # 特征提取器
        self.agent_encoder = nn.Linear(2, 64)
        self.target_encoder = nn.Linear(3, 64)
        self.obstacle_encoder = nn.Linear(4, 64)
        self.task_encoder = nn.Linear(4, 64)
        
        # 注意力层
        self.target_attention = MultiHeadAttention(64, config.attention_dim, config.num_heads)
        self.obstacle_attention = MultiHeadAttention(64, config.attention_dim, config.num_heads)
        
        # 融合层
        fusion_dim = 64 + config.attention_dim * 2 + 64  # agent + target_attn + obstacle_attn + task
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU()
        )
        
        # 输出头
        self.actor = nn.Linear(config.hidden_dim, 5)  # 5个动作
        self.critic = nn.Linear(config.hidden_dim, 1)
        
    def forward(self, obs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        返回: (action_logits, value)
        """
        batch_size = obs['agent_pos'].shape[0]
        
        # 编码各种特征
        agent_feat = F.relu(self.agent_encoder(obs['agent_pos']))  # [batch, 64]
        
        # 目标特征 + 注意力
        target_feat = F.relu(self.target_encoder(obs['targets']))  # [batch, max_targets, 64]
        target_mask = (obs['targets'][:, :, 0] >= 0)  # 有效目标掩码
        target_attended = self.target_attention(target_feat, target_mask)  # [batch, max_targets, attention_dim]
        target_pooled = torch.mean(target_attended, dim=1)  # [batch, attention_dim]
        
        # 障碍物特征 + 注意力
        obstacle_feat = F.relu(self.obstacle_encoder(obs['obstacles']))  # [batch, max_obstacles, 64]
        obstacle_mask = (obs['obstacles'][:, :, 0] >= 0)  # 有效障碍物掩码
        obstacle_attended = self.obstacle_attention(obstacle_feat, obstacle_mask)  # [batch, max_obstacles, attention_dim]
        obstacle_pooled = torch.mean(obstacle_attended, dim=1)  # [batch, attention_dim]
        
        # 任务特征
        task_feat = F.relu(self.task_encoder(obs['task_info']))  # [batch, 64]
        
        # 特征融合
        fused_feat = torch.cat([agent_feat, target_pooled, obstacle_pooled, task_feat], dim=1)
        hidden = self.fusion(fused_feat)  # [batch, hidden_dim]
        
        # 输出
        action_logits = self.actor(hidden)
        value = self.critic(hidden)
        
        return action_logits, value

# 4. PPO算法实现

class PPOAgent:
    """PPO智能体"""
    
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device(config.device)
        
        # 网络
        self.policy_net = AttentionPolicyNetwork(config).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=config.lr)
        
        # 经验缓冲区
        self.experiences = []
        
        # 统计信息
        self.episode_rewards = []
        self.episode_lengths = []
        self.policy_losses = []
        self.value_losses = []
        
    def select_action(self, obs: Dict[str, np.ndarray], training: bool = True) -> Tuple[int, float, float]:
        """选择动作"""
        # 转换为tensor
        obs_tensor = {}
        for key, value in obs.items():
            if isinstance(value, np.ndarray):
                obs_tensor[key] = torch.FloatTensor(value).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action_logits, value = self.policy_net(obs_tensor)
            
        # 采样动作
        dist = Categorical(logits=action_logits)
        if training:
            action = dist.sample()
        else:
            action = torch.argmax(action_logits, dim=-1)
        
        log_prob = dist.log_prob(action)
        
        return action.item(), log_prob.item(), value.item()
    
    def store_experience(self, state: Dict, action: int, reward: float, 
                        next_state: Dict, done: bool, log_prob: float, value: float):
        """存储经验"""
        experience = Experience(state, action, reward, next_state, done, log_prob, value)
        self.experiences.append(experience)
    
    def update(self) -> Dict[str, float]:
        """PPO更新"""
        if len(self.experiences) < self.config.batch_size:
            return {}
        
        # 计算回报和优势
        returns, advantages = self._compute_gae()
        
        # 准备训练数据
        states, actions, old_log_probs, old_values = self._prepare_training_data()
        
        # PPO更新
        total_policy_loss = 0
        total_value_loss = 0
        
        for epoch in range(self.config.update_epochs):
            # 随机打乱数据
            indices = torch.randperm(len(states['agent_pos']))
            
            for start_idx in range(0, len(indices), self.config.batch_size):
                end_idx = min(start_idx + self.config.batch_size, len(indices))
                batch_indices = indices[start_idx:end_idx]
                
                # 批次数据
                batch_states = {key: value[batch_indices] for key, value in states.items()}
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]
                
                # 前向传播
                action_logits, values = self.policy_net(batch_states)
                dist = Categorical(logits=action_logits)
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()
                
                # PPO损失
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.config.eps_clip, 1 + self.config.eps_clip) * batch_advantages
                
                policy_loss = -torch.min(surr1, surr2).mean() - 0.01 * entropy
                value_loss = F.mse_loss(values.squeeze(), batch_returns)
                
                total_loss = policy_loss + 0.5 * value_loss
                
                # 反向传播
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.5)
                self.optimizer.step()
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
        
        # 清空经验缓冲区
        self.experiences.clear()
        
        # 记录损失
        avg_policy_loss = total_policy_loss / (self.config.update_epochs * len(range(0, len(indices), self.config.batch_size)))
        avg_value_loss = total_value_loss / (self.config.update_epochs * len(range(0, len(indices), self.config.batch_size)))
        
        self.policy_losses.append(avg_policy_loss)
        self.value_losses.append(avg_value_loss)
        
        return {
            'policy_loss': avg_policy_loss,
            'value_loss': avg_value_loss,
            'experiences': len(self.experiences)
        }
    
    def _compute_gae(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """计算广义优势估计(GAE)"""
        returns = []
        advantages = []
        
        # 逆序计算
        next_value = 0
        next_advantage = 0
        
        for exp in reversed(self.experiences):
            if exp.done:
                next_value = 0
                next_advantage = 0
            
            delta = exp.reward + self.config.gamma * next_value - exp.value
            advantage = delta + self.config.gamma * 0.95 * next_advantage  # GAE lambda=0.95
            
            returns.insert(0, advantage + exp.value)
            advantages.insert(0, advantage)
            
            next_value = exp.value
            next_advantage = advantage
        
        returns = torch.FloatTensor(returns).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        
        # 标准化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return returns, advantages
    
    def _prepare_training_data(self) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
        """准备训练数据"""
        # 收集所有状态
        states = {key: [] for key in self.experiences[0].state.keys()}
        for exp in self.experiences:
            for key, value in exp.state.items():
                states[key].append(value)
        
        # 转换为tensor
        for key in states:
            states[key] = torch.FloatTensor(np.array(states[key])).to(self.device)
        
        actions = torch.LongTensor([exp.action for exp in self.experiences]).to(self.device)
        old_log_probs = torch.FloatTensor([exp.log_prob for exp in self.experiences]).to(self.device)
        old_values = torch.FloatTensor([exp.value for exp in self.experiences]).to(self.device)
        
        return states, actions, old_log_probs, old_values

# 5. 课程学习管理器

class CurriculumManager:
    """课程学习管理器"""
    
    def __init__(self, config: Config):
        self.config = config
        self.current_step = 0
        self.success_history = deque(maxlen=100)
        
    def get_difficulty(self) -> float:
        """根据训练进度和成功率自适应调整难度"""
        # 基础难度（基于训练步数）
        progress = min(self.current_step / self.config.curriculum_steps, 1.0)
        base_difficulty = 0.2 + 0.8 * progress
        
        # 根据最近成功率调整
        if len(self.success_history) > 10:
            recent_success = np.mean(list(self.success_history)[-10:])
            if recent_success > 0.8:
                base_difficulty = min(1.0, base_difficulty + 0.1)
            elif recent_success < 0.3:
                base_difficulty = max(0.1, base_difficulty - 0.1)
        
        return base_difficulty
    
    def update(self, success: bool):
        """更新课程学习状态"""
        self.current_step += 1
        self.success_history.append(1.0 if success else 0.0)

# 6. 训练和评估
class Trainer:
    """训练管理器"""
    
    def __init__(self, config: Config):
        self.config = config
        self.env = DynamicNavigationEnv(config)
        self.agent = PPOAgent(config)
        self.curriculum = CurriculumManager(config)
        
        # 统计信息
        self.episode_rewards = []
        self.episode_lengths = []
        self.success_rates = []
        self.difficulties = []
        
    def train(self, total_episodes: int = 2000):
        """训练智能体"""
        print(f"开始训练，总回合数: {total_episodes}")
        print(f"使用设备: {self.config.device}")
        
        for episode in range(total_episodes):
            # 获取当前难度
            difficulty = self.curriculum.get_difficulty()
            
            # 重置环境
            obs, info = self.env.reset(options={'difficulty': difficulty})
            
            episode_reward = 0
            episode_length = 0
            
            while True:
                # 选择动作
                action, log_prob, value = self.agent.select_action(obs, training=True)
                
                # 执行动作
                next_obs, reward, done, truncated, next_info = self.env.step(action)
                
                # 存储经验
                self.agent.store_experience(obs, action, reward, next_obs, done or truncated, log_prob, value)
                
                episode_reward += reward
                episode_length += 1
                
                obs = next_obs
                
                if done or truncated:
                    break
            
            # 更新课程学习
            success = info.get('current_target_idx', 0) >= info.get('total_targets', 1)
            self.curriculum.update(success)
            
            # 记录统计信息
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            self.difficulties.append(difficulty)
            
            # PPO更新
            if (episode + 1) % 10 == 0:  # 每10回合更新一次
                update_info = self.agent.update()
                
                if update_info:
                    print(f"回合 {episode+1}: 奖励={episode_reward:.2f}, "
                          f"长度={episode_length}, 难度={difficulty:.2f}, "
                          f"策略损失={update_info['policy_loss']:.4f}")
            
            # 定期评估
            if (episode + 1) % 100 == 0:
                success_rate = self.evaluate(num_episodes=10)
                self.success_rates.append(success_rate)
                print(f"回合 {episode+1} 评估: 成功率={success_rate:.2f}")
            
            # 保存模型
            if (episode + 1) % self.config.save_interval == 0:
                self.save_model(f"model_episode_{episode+1}.pth")
        
        print("训练完成！")
    
    def evaluate(self, num_episodes: int = 50, render: bool = False) -> float:
        """评估智能体性能"""
        successes = 0
        total_rewards = []
        
        for episode in range(num_episodes):
            obs, info = self.env.reset(options={'difficulty': 1.0})  # 使用最高难度评估
            episode_reward = 0
            
            while True:
                action, _, _ = self.agent.select_action(obs, training=False)
                obs, reward, done, truncated, info = self.env.step(action)
                episode_reward += reward
                
                if render and episode == 0:  # 只渲染第一个回合
                    self.env.render()
                
                if done or truncated:
                    if info.get('current_target_idx', 0) >= info.get('total_targets', 1):
                        successes += 1
                    total_rewards.append(episode_reward)
                    break
        
        success_rate = successes / num_episodes
        avg_reward = np.mean(total_rewards)
        
        print(f"评估结果: 成功率={success_rate:.2f}, 平均奖励={avg_reward:.2f}")
        return success_rate
    
    def save_model(self, filename: str):
        """保存模型"""
        torch.save({
            'policy_net': self.agent.policy_net.state_dict(),
            'optimizer': self.agent.optimizer.state_dict(),
            'config': self.config,
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'success_rates': self.success_rates,
        }, filename)
        print(f"模型已保存到: {filename}")
    
    def load_model(self, filename: str):
        """加载模型"""
        checkpoint = torch.load(filename, map_location=self.config.device)
        self.agent.policy_net.load_state_dict(checkpoint['policy_net'])
        self.agent.optimizer.load_state_dict(checkpoint['optimizer'])
        self.episode_rewards = checkpoint.get('episode_rewards', [])
        self.episode_lengths = checkpoint.get('episode_lengths', [])
        self.success_rates = checkpoint.get('success_rates', [])
        print(f"模型已从 {filename} 加载")

# 7. 可视化和分析

class Visualizer:
    """可视化工具"""
    
    @staticmethod
    def plot_training_curves(trainer: Trainer):
        """绘制训练曲线"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 回合奖励
        axes[0, 0].plot(trainer.episode_rewards)
        axes[0, 0].set_title('回合奖励')
        axes[0, 0].set_xlabel('回合')
        axes[0, 0].set_ylabel('奖励')
        
        # 回合长度
        axes[0, 1].plot(trainer.episode_lengths)
        axes[0, 1].set_title('回合长度')
        axes[0, 1].set_xlabel('回合')
        axes[0, 1].set_ylabel('步数')
        
        # 成功率
        if trainer.success_rates:
            x_success = list(range(100, len(trainer.episode_rewards) + 1, 100))[:len(trainer.success_rates)]
            axes[1, 0].plot(x_success, trainer.success_rates)
            axes[1, 0].set_title('成功率')
            axes[1, 0].set_xlabel('回合')
            axes[1, 0].set_ylabel('成功率')
        
        # 难度曲线
        axes[1, 1].plot(trainer.difficulties)
        axes[1, 1].set_title('难度变化')
        axes[1, 1].set_xlabel('回合')
        axes[1, 1].set_ylabel('难度')
        
        plt.tight_layout()
        plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    @staticmethod
    def plot_loss_curves(agent: PPOAgent):
        """绘制损失曲线"""
        if not agent.policy_losses:
            print("没有损失数据可绘制")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        ax1.plot(agent.policy_losses)
        ax1.set_title('策略损失')
        ax1.set_xlabel('更新次数')
        ax1.set_ylabel('损失')
        
        ax2.plot(agent.value_losses)
        ax2.set_title('价值损失')
        ax2.set_xlabel('更新次数')
        ax2.set_ylabel('损失')
        
        plt.tight_layout()
        plt.savefig('loss_curves.png', dpi=300, bbox_inches='tight')
        plt.show()

# 8. 主程序
def main():
    """主程序"""
    print("强化学习项目：带注意力机制的多任务导航智能体")
    print("=" * 70)
    
    # 创建配置
    config = Config()
    print(f"配置信息:")
    print(f"  设备: {config.device}")
    print(f"  网格大小: {config.grid_size}x{config.grid_size}")
    print(f"  隐藏层维度: {config.hidden_dim}")
    print(f"  注意力维度: {config.attention_dim}")
    print()
    
    # 创建训练器
    trainer = Trainer(config)
    
    # 模式选择
    mode = input("选择模式 [train/evaluate/demo]: ").strip().lower()
    
    if mode == "train":
        # 训练模式
        episodes = int(input("输入训练回合数 (默认2000): ") or "2000")
        print(f"\n开始训练 {episodes} 回合...")
        
        trainer.train(total_episodes=episodes)
        
        # 绘制训练曲线
        print("\n生成训练曲线...")
        Visualizer.plot_training_curves(trainer)
        Visualizer.plot_loss_curves(trainer.agent)
        
        # 最终评估
        print("\n最终评估...")
        final_success_rate = trainer.evaluate(num_episodes=100)
        print(f"最终成功率: {final_success_rate:.2f}")
        
    elif mode == "evaluate":
        # 评估模式
        model_path = input("输入模型路径 (默认model_episode_2000.pth): ").strip()
        if not model_path:
            model_path = "model_episode_2000.pth"
        
        if os.path.exists(model_path):
            trainer.load_model(model_path)
            trainer.evaluate(num_episodes=100, render=False)
        else:
            print(f"模型文件 {model_path} 不存在")
    
    elif mode == "demo":
        # 演示模式
        print("\n运行演示...")
        obs, info = trainer.env.reset(options={'difficulty': 0.8})
        
        for step in range(100):
            action, _, _ = trainer.agent.select_action(obs, training=False)
            obs, reward, done, truncated, info = trainer.env.step(action)
            
            print(f"步骤 {step+1}: 动作={action}, 奖励={reward:.2f}, "
                  f"目标进度={info['current_target_idx']}/{info['total_targets']}")
            
            if done or truncated:
                success = info['current_target_idx'] >= info['total_targets']
                print(f"回合结束! {'成功' if success else '失败'}")
                break
    
    else:
        print("无效模式，程序退出")

if __name__ == "__main__":
    main()
