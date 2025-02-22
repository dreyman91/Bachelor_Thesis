from pettingzoo.utils import BaseWrapper
import numpy as np


class LimitedObservabilityWrapper(BaseWrapper): # Inherits from pettingzoo's wrapper
    def __init__(self, env, hide_agents=False, hide_landmarks=False):
        super().__init__(env) # Initialize Environment
        self.hide_agents = hide_agents # Boolean to control agents visibility
        self.hide_landmarks = hide_landmarks # Boolean to control landmark visibility

    # Overiding reset method
    def reset(self, seed=None, options=None):
        observations, _ = self.env.reset(seed=seed, options=options) # call original reset
        return self.modify_observations(observations)

    # Overiding Step method
    def step(self, actions):
        observations, rewards, terminations, truncations, infos = self.env.step(actions) #Execute step
        return self.modify_observations(observations), rewards, terminations, truncations, infos

    # Modifying Observations dynamically
    def modify_observations(self, observations):
        modified_obs = {} # Create dictionary for modified observations
        for agent, agent_obs in observations.items(): # Loop through each agent's observation
            obs_array = np.array(agent_obs) # Convert to numpy array

            if self.hide_agents:
                obs_array[-8:] = 0  # Zero out the last 8 values i.e other agent's position

            if self.hide_landmarks:
                obs_array[4:10] = 0 # Zero out 6 landmark positions

            modified_obs[agent] = obs_array # Store modified observation
        return modified_obs




