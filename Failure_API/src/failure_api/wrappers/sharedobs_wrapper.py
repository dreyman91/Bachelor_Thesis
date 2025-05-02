from pettingzoo.utils.wrappers import BaseWrapper
from gymnasium import spaces


class SharedObsWrapper(BaseWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.agents = env.possible_agents

    def observe(self, agent):
        return {
            other_agent: self.env.observe(other_agent)
            for other_agent in self.agents
            if other_agent in self.env.agents  # Only active agents
        }

    def last(self, observe: bool = True):
        agent = self.env.agent_selection
        if agent not in self.env.agents:
            return None, 0.0, True, False, {}
        _, rew, term, trunc, info = self.env.last()
        if not observe:
            return None, rew, term, trunc, info
        current_agent = self.env.agent_selection

        if current_agent not in self.env.agents:
            return None, rew, term, trunc, info

        shared_obs = self.observe(current_agent)
        return shared_obs, rew, term, trunc, info

    def observation_space(self, agent):
        return spaces.Dict({
            other_agent: self.env.observation_space(other_agent)
            for other_agent in self.agents
            if other_agent in self.env.agents
        })

    def reset(self, seed=None, options=None):
        return self.env.reset(seed=seed, options=options)
