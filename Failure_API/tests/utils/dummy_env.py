
from pettingzoo.utils import AECEnv
from gymnasium import spaces
import numpy as np

class DummyEnv(AECEnv):
    def __init__(self):
        self.possible_agents = ["agent0", "agent1"]
        self.agents = self.possible_agents[:]
        self.agent_selection = self.agents[0]

    def observe(self, agent):
        obs = {
            "agent0": np.array([3, -1], dtype=np.float32),
            "agent1": np.array([0, 4], dtype=np.float32),
        }
        return obs[agent]

    def action_space(self, agent):
        return spaces.Discrete(4)

    def observation_space(self, agent):
        return spaces.Box(low=-10, high=10, shape=(2,), dtype=np.float32)

    def step(self, action):
        self.agent_selection = self.agents[1 - self.agents.index(self.agent_selection)]

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
        self.agent_selection = self.agents[0]