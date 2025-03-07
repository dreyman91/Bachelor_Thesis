import numpy as np
import time
import cProfile
import pstats
import io
from memory_profiler import profile
from Wrapper.comm_failure_api import CommunicationFailure

# Dummy PettingZoo-like environment for performance testing
class LargeScaleDummyEnv:
    def __init__(self, num_agents=10):
        self.possible_agents = [f"agent_{i}" for i in range(num_agents)]
        self._observation_space = {
            agent: np.random.random((5,)) for agent in self.possible_agents
        }
        self._action_space = {
            agent: np.random.randint(0, 5) for agent in self.possible_agents
        }


    def observation_space(self, agent):
        return self._observation_space[agent]

    def action_space(self, agent):
        return self._action_space[agent]

    def reset(self, seed=None, options=None):
        obs = {agent: np.random.random((5,)) for agent in self.possible_agents}
        infos = {agent: {} for agent in self.possible_agents}
        return obs, infos

    def step(self, actions):
        obs = {agent: np.random.random((5,)) for agent in self.possible_agents}
        rewards = {agent: np.random.random() for agent in self.possible_agents}
        terminations = {agent: np.random.choice([True, False]) for agent in self.possible_agents}
        truncations = {agent: np.random.choice([True, False]) for agent in self.possible_agents}
        infos = {agent: {} for agent in self.possible_agents}
        return obs, rewards, terminations, truncations, infos