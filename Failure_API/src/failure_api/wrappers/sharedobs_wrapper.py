from .base_wrapper import BaseWrapper
from gymnasium import spaces
import time


class SharedObsWrapper(BaseWrapper):
    def __init__(self, env):
        super().__init__(env)
        self._obs_cache = {}
        self._last_step = -1
        self._step_count = 0
        self._cached_agents = None
        self._last_agent = None
        self._last_result = None

    @property
    def agents(self):
        return self.env.agents

    def step(self, action):
        self._step_count += 1
        self._cached_agents = None
        self._last_agent = None
        return self.env.step(action)

    def observe(self, agent):
        if self._step_count != self._last_step:
            self.observe_all()


        if agent in self._obs_cache:
            return self._obs_cache.copy()
        return None

    def observe_all(self):
        """Compute all agent observations once per step."""
        self._obs_cache.clear()
        self._last_step = self._step_count
        for agent in self.agents:
            self._obs_cache[agent] = self.env.observe(agent)
        return self._obs_cache

    def last(self, observe=True):
        agent = self.env.agent_selection

        # Cache result
        if agent == self._last_agent and self._last_result is not None:
            return self._last_result

        if agent not in self.env.agents:
            result = (None, 0.0, True, False, {})
        else:
            _, rew, term, trunc, info = self.env.last()
            result = (self.observe(agent) if observe else None, rew, term, trunc, info)

        self._last_agent = agent
        self._last_result = result
        return result

    def reset(self, seed=None, options=None):
        result = self.env.reset(seed=seed, options=options)
        self._obs_cache.clear()
        self._last_step = -1
        self._step_count = 0
        self._cached_agents = None
        self._last_agent = None
        self._last_result = None
        return result
