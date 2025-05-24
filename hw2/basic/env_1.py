""" 我们将实现一个非常简单的游戏，名为GridWorldEnv，
它由一个固定大小的二维方格网格组成。
代理可以在每个时间步长内垂直或水平移动于网格单元之间，其目标是导航至在回合开始时随机放置在网格上的目标。

游戏基本信息

观察提供了目标和代理的位置。

我们的环境中有 4 个离散动作，分别对应"右"、"上"、"左"和"下"的运动。

当代理导航到目标所在的网格单元时，环境结束（终止）。

代理只有达到目标时才会获得奖励，即当代理达到目标时奖励为一，否则奖励为零。 """

# 我们的自定义环境 inherit gymnasium.Env,需要定义的空间包括观察和行动空间

from typing import Optional
import numpy as np
import gymnasium as gym

class GridWorldEnv(gym.Env):
    def __init__(self, size:int=5):
        self.size = size

        self._agent_location = np.array([-1,-1],dtype=np.int32)
        self._target_location = np.array([-1,-1],dtype=np.int32)

        self.observation_space = gym.spaces.Dict(
            {
                "agent":gym.spaces.Box(0, size-1,shape=(2,),dtype=int),
                "target":gym.spaces.Box(0, size-1,shape=(2,),dtype=int)
            }
        )

        self.action_space = gym.spaces.Discrete(4)

        self._action_to_direction = {
            0:np.array([1,0]),
            1:np.array([0,1]),
            2:np.array([-1,0]),
            3:np.array([0,-1])
        }
        
    def _get_obs(self):
        return {"agent":self._agent_location, "target":self._target_location}

    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location, ord = 1
            )
        }

    # reset function:seed & options
    def reset(self, seed:Optional[int] = None, options:Optional[dict] = None):
        super().reset(seed=seed)

        self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)

        self._target_location = self._agent_location
        while np.array_equal(self._target_location, self._agent_location):
            self._target_location = self.np_random.integers(0, self.size, size=2, dtype=int)

        observation = self._get_obs()
        info = self._get_info()

        return observation, info
        
    # step function : accepts an action and computes the state of the environment after applying the action, returning the tuple (next observation, reward, if terminated, if truncated)
    # self._actin_to_direction ; exam reward ; _get_obs ; _get_info

    def step(self, action):
        direction = self._action_to_direction[action]
        self._agent_location = np.clip(
            self._agent_location + direction, 0, self.size-1
        )
        terminated = np.array_equal(self._agent_location, self._target_location)
        truncated = False
        reward = 1 if terminated else 0
        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

# register the environment
gym.register(
    id = "gymnasium_env/Gridworld-v114514",
    entry_point=GridWorldEnv
)

