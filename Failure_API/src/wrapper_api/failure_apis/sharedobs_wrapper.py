from pettingzoo.utils import BaseWrapper


class SharedObsWrapper(BaseWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.agents = env.possible_agents

    def observe(self, agent):
        return {
            other_agent: self.env.observe(other_agent)
            for other_agent in self.agents
            if other_agent in self.env.agents # Only active agents
        }