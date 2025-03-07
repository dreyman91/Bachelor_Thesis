from pettingzoo import AECEnv
import numpy as np
import pytest
from gymnasium import spaces
from Wrapper.comm_failure_api import CommunicationFailure  # Adjust import as needed
from pettingzoo.utils.env import AECEnv, ParallelEnv


class DummyParallelEnv(ParallelEnv):
    """ A simple Parallel environment to test communication failures """

    def __init__(self, possible_agents=None):
        self.possible_agents = possible_agents or ["agent_0", "agent_1", "agent_2"]
        self.action_spaces = {agent: spaces.Discrete(3) for agent in self.possible_agents}
        self.observation_spaces = {agent: spaces.Box(low=0, high=1, shape=(3,), dtype=np.float32)
                                   for agent in self.possible_agents}

    def reset(self, seed=None, options=None):
        obs = {agent: np.ones(3, dtype=np.float32) for agent in self.possible_agents}
        infos = {agent: {} for agent in self.possible_agents}
        print(f"DummyParallelEnv.reset() called -> Returning {len(obs)} observations and {len(infos)} infos")
        return obs, infos

    def step(self, actions):
        print(f"DummyParallelEnv.step() called with actions: {actions}")

        obs = {agent: np.ones(3, dtype=np.float32) for agent in self.possible_agents}
        rewards = {agent: 0.0 for agent in self.possible_agents}
        terminations = {agent: False for agent in self.possible_agents}
        truncations = {agent: False for agent in self.possible_agents}
        infos = {agent: {} for agent in self.possible_agents}

        print(f"DummyParallelEnv.step() returning: "
              f"{len(obs)} observations, "
              f"{len(rewards)} rewards, "
              f"{len(terminations)} terminations, "
              f"{len(truncations)} truncations, "
              f"{len(infos)} infos")

        return obs, rewards, terminations, truncations, infos


class DummyAECEnv(AECEnv):
    """ A simple AEC environment to test communication failures """

    def __init__(self, possible_agents=None):
        super().__init__()
        self.possible_agents = possible_agents or ["agent_0", "agent_1", "agent_2"]
        self.agent_order = self.possible_agents.copy()
        self.action_spaces = {agent: spaces.Discrete(3) for agent in self.possible_agents}
        self.observation_spaces = {agent: spaces.Box(low=0, high=1, shape=(3,), dtype=np.float32)
                                   for agent in self.possible_agents}

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
        self._agent_selector = iter(self.agent_order)
        self.agent_selection = next(self._agent_selector)

        obs = {agent: np.ones(3, dtype=np.float32) for agent in self.agents}
        infos = {agent: {} for agent in self.agents}

        print(f"DummyAECEnv.reset() called -> Returning {len(obs)} observations and {len(infos)} infos")

        return obs, infos

    def step(self, action):
        print(f"DummyAECEnv.step() called with action: {action}")

        obs = {agent: np.ones(3, dtype=np.float32) for agent in self.agents}
        rewards = {agent: 0.0 for agent in self.agents}
        terminations = {agent: False for agent in self.agents}
        truncations = {agent: False for agent in self.agents}
        infos = {agent: {} for agent in self.agents}

        print(f"DummyAECEnv.step() returning: "
              f"{len(obs)} observations, "
              f"{len(rewards)} rewards, "
              f"{len(terminations)} terminations, "
              f"{len(truncations)} truncations, "
              f"{len(infos)} infos")

        # Move to the next agent
        try:
            self.agent_selection = next(self._agent_selector)
        except StopIteration:
            self.agent_selection = None

        return obs, rewards, terminations, truncations, infos
