# Gymnasium 使用指南

## 基本流程

### 1. 创建环境
```python
env = gym.make("LunarLander-v3", render_mode="human")
```
- 使用 `gym.make()` 创建环境
- `render_mode` 指定可视化方式

### 2. 重置环境
```python
observation, info = env.reset()
```
- 获取初始观察值和信息
- 可选参数：`seed`（随机种子）、`options`（环境配置）

### 3. 交互循环
```python
while not episode_over:
    action = env.action_space.sample()  # 选择动作
    observation, reward, terminated, truncated, info = env.step(action)
    episode_over = terminated or truncated
```

### 4. 关键概念
- **observation**: 环境状态观察
- **action**: 智能体执行的动作
- **reward**: 执行动作后获得的奖励
- **terminated**: 环境自然结束（成功/失败）
- **truncated**: 环境被强制结束（超时等）

### 5. 清理资源
```python
env.close()
```


## 动作和观察空间详解

### 空间的基本概念

每个环境都通过 `action_space` 和 `observation_space` 属性指定有效动作和观察的格式。这些空间定义了：
- **action_space**: 智能体可以执行的所有有效动作
- **observation_space**: 环境可能返回的所有有效观察

### Space 类的核心功能

所有空间都是 `Space` 类的实例，提供两个关键方法：
- **Space.contains()**: 检查给定值是否属于该空间
- **Space.sample()**: 从空间中随机采样一个有效值

```python
# 示例用法
action = env.action_space.sample()  # 随机动作
is_valid = env.action_space.contains(action)  # 检查动作是否有效
```

### 空间类型详解

#### 1. Box
描述有上下界限的连续空间，可以是任意n维形状
```python
# 例如：3D位置坐标，每个维度范围在[-1, 1]
Box(low=-1.0, high=1.0, shape=(3,))
```

#### 2. Discrete
描述离散空间，值域为 {0, 1, ..., n-1}
```python
# 例如：4个可能的动作 {0, 1, 2, 3}
Discrete(4)
```

#### 3. MultiBinary
描述任意n维形状的二进制空间
```python
# 例如：3个开关状态，每个可以是0或1
MultiBinary(3)  # 可能值：[0,0,0], [1,0,1], [1,1,1] 等
```

#### 4. MultiDiscrete
由多个不同大小的离散空间组成
```python
# 例如：第一个动作有3种选择，第二个动作有5种选择
MultiDiscrete([3, 5])  # 可能值：[0,0], [2,4], [1,2] 等
```

#### 5. Text
描述具有最小和最大长度的字符串空间
```python
# 例如：长度在5-20之间的文本
Text(min_length=5, max_length=20)
```

#### 6. Dict
描述由简单空间组成的字典
```python
# 例如：包含位置和速度信息的观察
Dict({
    'position': Box(low=-1, high=1, shape=(2,)),
    'velocity': Box(low=-10, high=10, shape=(2,))
})
```

#### 7. Tuple
描述简单空间组成的元组
```python
# 例如：位置和离散动作的组合
Tuple((Box(low=-1, high=1, shape=(2,)), Discrete(4)))
```

#### 8. Graph（特殊用途）
描述数学图（网络），包含相互连接的节点和边

#### 9. Sequence（特殊用途）
描述由简单空间元素组成的可变长度序列

### 实际应用示例

```python
import gymnasium as gym

env = gym.make("LunarLander-v3")

# 查看空间信息
print("动作空间:", env.action_space)
print("观察空间:", env.observation_space)

# 采样和验证
action = env.action_space.sample()
print("随机动作:", action)
print("动作是否有效:", env.action_space.contains(action))

observation, _ = env.reset()
print("观察形状:", observation.shape)
print("观察是否有效:", env.observation_space.contains(observation))
```


## 环境修改与包装器

### 包装器的基本概念

包装器（Wrappers）是一种便捷的方式来修改现有环境，无需直接修改底层代码。使用包装器的优势：

- **避免重复代码**：减少样板代码的编写
- **模块化设计**：使环境修改更加模块化
- **链式组合**：可以将多个包装器组合使用
- **默认包装**：大多数通过 `gymnasium.make()` 生成的环境已经默认使用了一些包装器

### 包装器的使用方法

#### 基本使用步骤
1. 首先初始化基础环境
2. 将环境传递给包装器的构造函数

```python
import gymnasium as gym
from gymnasium.wrappers import FlattenObservation

# 创建基础环境
env = gym.make("CarRacing-v3")
print("原始观察空间形状:", env.observation_space.shape)  # (96, 96, 3)

# 应用包装器
wrapped_env = FlattenObservation(env)
print("包装后观察空间形状:", wrapped_env.observation_space.shape)  # (27648,)
```

### 常用包装器类型

#### 1. TimeLimit
限制最大时间步数，超时后发出 `truncated` 信号
```python
from gymnasium.wrappers import TimeLimit
env = TimeLimit(base_env, max_episode_steps=1000)
```

#### 2. ClipAction
将传递给 `step()` 的动作裁剪到环境动作空间范围内
```python
from gymnasium.wrappers import ClipAction
env = ClipAction(base_env)
```

#### 3. RescaleAction
对动作应用仿射变换，线性缩放到新的上下界
```python
from gymnasium.wrappers import RescaleAction
env = RescaleAction(base_env, min_action=-2.0, max_action=2.0)
```

#### 4. TimeAwareObservation
在观察中添加时间步索引信息，有助于确保转换的马尔可夫性
```python
from gymnasium.wrappers import TimeAwareObservation
env = TimeAwareObservation(base_env)
```

#### 5. FlattenObservation
将多维观察空间展平为一维数组
```python
from gymnasium.wrappers import FlattenObservation
env = FlattenObservation(base_env)
```

#### 6. 其他常用包装器
- **FrameStack**: 堆叠多个连续帧
- **AtariPreprocessing**: Atari游戏专用预处理
- **RecordVideo**: 录制环境视频
- **RecordEpisodeStatistics**: 记录回合统计信息

### 包装器链式组合

可以将多个包装器串联使用：

```python
import gymnasium as gym
from gymnasium.wrappers import TimeLimit, ClipAction, FlattenObservation

# 创建基础环境
base_env = gym.make("CarRacing-v3")

# 链式应用多个包装器
env = base_env
env = TimeLimit(env, max_episode_steps=1000)
env = ClipAction(env)
env = FlattenObservation(env)

print("最终包装器链:", env)
```

### 获取原始环境

使用 `unwrapped` 属性可以获取所有包装器层下面的原始环境：

```python
# 查看包装器层次结构
print("包装器链:", wrapped_env)
# 输出: <FlattenObservation<TimeLimit<OrderEnforcing<PassiveEnvChecker<CarRacing<CarRacing-v3>>>>>>

# 获取原始环境
original_env = wrapped_env.unwrapped
print("原始环境:", original_env)
# 输出: <gymnasium.envs.box2d.car_racing.CarRacing object at 0x...>
```

### 自定义包装器

可以通过继承 `gymnasium.Wrapper` 创建自定义包装器：

```python
import gymnasium as gym
from gymnasium import Wrapper

class CustomWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        # 自定义初始化
    
    def step(self, action):
        # 自定义步进逻辑
        obs, reward, terminated, truncated, info = self.env.step(action)
        # 可以修改观察、奖励等
        return obs, reward, terminated, truncated, info
    
    def reset(self, **kwargs):
        # 自定义重置逻辑
        return self.env.reset(**kwargs)

# 使用自定义包装器
env = gym.make("CartPole-v1")
custom_env = CustomWrapper(env)
```

### 实际应用示例

```python
import gymnasium as gym
from gymnasium.wrappers import (
    TimeLimit, 
    ClipAction, 
    FlattenObservation,
    RecordEpisodeStatistics
)

def create_wrapped_environment(env_name, max_steps=1000):
    """创建一个完整包装的环境"""
    
    # 基础环境
    env = gym.make(env_name)
    
    # 添加时间限制
    env = TimeLimit(env, max_episode_steps=max_steps)
    
    # 裁剪动作
    env = ClipAction(env)
    
    # 展平观察（如果是多维的）
    if len(env.observation_space.shape) > 1:
        env = FlattenObservation(env)
    
    # 记录统计信息
    env = RecordEpisodeStatistics(env)
    
    return env

# 使用示例
env = create_wrapped_environment("CarRacing-v3", max_steps=2000)

# 运行一个回合
obs, info = env.reset()
for _ in range(100):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break

env.close()
```

---

### 核心概念：Q-Learning

在解释代码之前，我们先理解 **Q-Learning** 是什么。

* **目标**：Q-learning 的目标是学习一个叫做 **动作价值函数 (Action-Value Function)** 的东西，我们通常用 $Q(s, a)$ 来表示它。
* **含义**：$Q(s, a)$ 的意思是，在某个特定的**状态 (State)** $s$ 下，执行某个**动作 (Action)** $a$ 后，一直到游戏结束，预期能获得的总**奖励 (Reward)** 是多少。
* **策略**：如果我们拥有了一个完美的 $Q$ 函数（或者说一个存储了所有 $Q(s,a)$ 值的 Q-table），那么在任何状态 $s$ 下，我们只需要选择那个能使 $Q(s,a)$ 值最大的动作 $a$，这就是最优策略。
* **学习方式**：代理通过与环境互动来逐步优化它的 $Q$ 函数。它会不断地尝试，并根据得到的结果（奖励）来更新自己的 Q 值，让它越来越接近真实的期望奖励。

---

### 第一部分：构建代理 (`BlackjackAgent` 类)

这个类定义了我们代理的行为和“大脑”。

#### `__init__` (初始化函数)

这是创建代理实例时运行的函数，用于设置其所有基本属性。

```python
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
        # ...
```

* `env`: 这是代理所处的环境，也就是 `gymnasium` 库创建的二十一点游戏实例。代理需要通过它来获取状态、执行动作和获得奖励。
* `self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))`: 这是代理的“大脑”，即 Q-table。
    * `defaultdict`: 我们使用 `defaultdict` 是因为我们事先不知道会遇到多少种不同的游戏状态（玩家手牌总和、庄家明牌、是否有可用的 Ace）。如果一个状态第一次出现，`defaultdict` 会自动为它创建一个默认值。
    * `lambda: np.zeros(env.action_space.n)`: 这就是默认值。对于每个状态，我们都关联一个 NumPy 数组。数组的长度是 `env.action_space.n`（在二十一点中是 2，代表“不要牌(stand)”和“要牌(hit)”两个动作）。数组初始值为 `[0., 0.]`，表示在游戏开始时，代理对任何状态下采取任何动作的价值都一无所知。
* `self.lr` (learning\_rate): **学习率** $ \alpha $。它控制了代理每次学习的“步长”。一个较小的学习率意味着代理每次只根据新信息做微小的调整，学习过程更稳定但更慢。
* `self.discount_factor` (discount\_factor): **折扣因子** $ \gamma $。它衡量了未来奖励的重要性。一个接近 1 的值意味着代理是“有远见的”，非常看重未来的奖励。一个接近 0 的值意味着代理是“短视的”，只关心眼前的即时奖励。
* `self.epsilon`, `self.epsilon_decay`, `self.final_epsilon`: 这三个参数共同构成了 **Epsilon-Greedy 策略**。
    * `epsilon` ($ \epsilon $) 是探索的概率。
    * `epsilon_decay` 是每次游戏结束后 `epsilon` 的衰减值。
    * `final_epsilon` 是 `epsilon` 的最小值，确保代理永远不会完全停止探索。
* `self.training_error`: 一个列表，用于记录每次更新 Q 值时的“时序差分误差 (Temporal Difference Error)”，方便后续可视化分析。

#### `get_action` (选择动作)

这个函数实现了 **Epsilon-Greedy策略**，用于平衡**探索 (Exploration)** 和**利用 (Exploitation)**。

```python
    def get_action(self, obs: tuple[int, int, bool]) -> int:
        # with probability epsilon return a random action to explore the environment
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        # with probability (1 - epsilon) act greedily (exploit)
        else:
            return int(np.argmax(self.q_values[obs]))
```

* **探索**：以 $ \epsilon $ 的概率，代理会随机选择一个动作 (`env.action_space.sample()`)。这很重要，因为这样可以确保代理去尝试那些当前看起来不是最优的动作，也许会发现更好的策略。
* **利用**：以 $ 1 - \epsilon $ 的概率，代理会选择当前它认为最好的动作。它会查找当前状态 `obs` 对应的 Q 值数组 (`self.q_values[obs]`)，然后 `np.argmax` 会返回那个值最大的动作的索引（0 或 1）。

#### `update` (更新 Q 值)

这是 Q-Learning 算法最核心的部分。每次代理执行一个动作后，都会调用这个函数来学习。

```python
    def update(
        self,
        obs: tuple[int, int, bool],
        action: int,
        reward: float,
        terminated: bool,
        next_obs: tuple[int, int, bool],
    ):
        future_q_value = (not terminated) * np.max(self.q_values[next_obs])
        temporal_difference = (
            reward + self.discount_factor * future_q_value - self.q_values[obs][action]
        )
        self.q_values[obs][action] = (
            self.q_values[obs][action] + self.lr * temporal_difference
        )
```

这部分代码实现了 Q-Learning 的更新公式：
$Q_{new}(s, a) \leftarrow Q_{old}(s, a) + \alpha \cdot [R + \gamma \cdot \max_{a'} Q(s', a') - Q_{old}(s, a)]$

让我们逐行分解代码：
1.  `future_q_value = (not terminated) * np.max(self.q_values[next_obs])`:
    * `np.max(self.q_values[next_obs])`: 这就是公式中的 $ \max_{a'} Q(s', a') $。它代表了在到达**下一个状态 `next_obs`** 后，我们能期望的最好未来价值（即在那个新状态下采取最优动作能得到的 Q 值）。
    * `(not terminated)`: 这是一个关键细节。如果游戏在这一步**结束了 (`terminated` is True)**，那么就没有“未来”了，所以未来的价值就是 0。`not terminated` 在 Python 中会变成 1 (如果没结束) 或 0 (如果结束了)。

2.  `temporal_difference = (reward + self.discount_factor * future_q_value - self.q_values[obs][action])`:
    * 这就是公式中方括号里的部分，叫做**时序差分误差 (TD Error)**。
    * `reward + self.discount_factor * future_q_value`: 这是我们对 $Q(s, a)$ 的**新估计值**。它等于我们获得的**即时奖励** `reward` 加上经过折扣的**未来最佳价值**。
    * `- self.q_values[obs][action]`: 这是我们对 $Q(s, a)$ 的**旧估计值**。
    * TD Error 的意义是：我们新的、更准确的估计值与旧的估计值之间的差距有多大。这个差距也反映了我们预测的“惊喜程度”。

3.  `self.q_values[obs][action] = self.q_values[obs][action] + self.lr * temporal_difference`:
    * 这就是最终的更新步骤。我们把旧的 Q 值，加上一小部分 (由学习率 `lr` 控制) 的 TD Error，来得到新的 Q 值。这使得我们的 Q 值向着更准确的估计值迈进了一小步。

#### `decay_epsilon` (衰减 Epsilon)

```python
    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)
```

* 这个函数在每局游戏结束后被调用。
* 它会减少 `epsilon` 的值，使得代理随着训练的进行，越来越倾向于“利用”已学到的知识，而不是“探索”。
* `max(self.final_epsilon, ...)` 确保 `epsilon` 不会降到 `final_epsilon` 以下，保留一点点探索的可能性。

---

### 第二部分：训练代理

这部分代码设置了训练过程并执行它。

* **超参数 (Hyperparameters)**:
    * `learning_rate`: 学习率，传给代理。
    * `n_episodes`: 训练的总回合数（玩多少局游戏）。10万局对于这个问题来说是一个合理的数目。
    * `start_epsilon`: 初始探索率，设为 1.0 意味着一开始完全随机探索。
    * `epsilon_decay`: 衰减率。这里的计算方式 `start_epsilon / (n_episodes / 2)` 意味着 `epsilon` 会在大约一半的训练回合中从 1.0 衰减到接近 0。
    * `final_epsilon`: 最终探索率，设为 0.1。

* **训练循环 (Training Loop)**:
    ```python
    for episode in tqdm(range(n_episodes)):
        obs, info = env.reset() # 开始新的一局游戏，获取初始状态
        done = False

        while not done:
            action = agent.get_action(obs) # 代理根据当前状态选择动作
            next_obs, reward, terminated, truncated, info = env.step(action) # 在环境中执行动作，获取结果

            agent.update(obs, action, reward, terminated, next_obs) # 代理根据结果更新Q值

            done = terminated or truncated # 判断游戏是否结束
            obs = next_obs # 更新当前状态为下一步的状态

        agent.decay_epsilon() # 一局游戏结束，降低epsilon
    ```
    这个循环模拟了代理玩 `n_episodes` 局游戏的过程。在每一局游戏 (`while not done`) 的每一步，代理都会：**观察 -> 决策 -> 行动 -> 学习**。

---

### 第三部分：可视化训练过程

这部分代码使用 `matplotlib` 将训练过程中的关键指标画成图表，帮助我们判断代理是否在有效地学习。

* **`get_moving_avgs`**: 一个辅助函数，用于计算**移动平均值**。由于单局游戏的结果随机性很大，直接画图会看到非常杂乱的曲线。移动平均可以将数据平滑，更容易看出长期趋势。
* **Episode Rewards (每回合奖励)**:
    * 我们期望这个曲线**总体呈上升趋势**。在二十一点中，赢了奖励是+1，输了是-1，平局是0。由于庄家有固有优势，一个优秀的代理其平均回报也很难超过 0，但它应该会从一个很低的负值（比如-0.5，表示经常输）逐渐上升到接近-0.05的水平。
* **Episode Lengths (每回合长度)**:
    * 表示每局游戏持续了多少步（要了多少次牌）。这个指标的意义不如奖励直观。一个好的策略可能会导致游戏长度适中。
* **Training Error (训练误差)**:
    * 也就是我们之前存下来的 TD Error。我们期望这个曲线**总体呈下降趋势**并最终收敛。一开始，代理的 Q 值非常不准，所以 TD Error 会很大。随着学习的进行，它的预测越来越准，TD Error 也就越来越小。

这段代码完整地展示了如何从零开始构建、训练和评估一个 Q-learning 强化学习代理，来解决一个经典的游戏问题。它涵盖了状态、动作、奖励、Q-table、Epsilon-Greedy 探索策略以及核心的 Q-learning 更新法则等所有关键概念。