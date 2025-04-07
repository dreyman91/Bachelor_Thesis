import numpy as np
from pettingzoo import AECEnv
from gymnasium import spaces
from typing import Optional, Any
from abc import ABC, abstractmethod
from pettingzoo.utils import BaseWrapper

from Failure_API.src.wrapper_api.models.noise_model import NoiseModel


class NoiseWrapper(BaseWrapper):
    def __init__(self,
                 env: AECEnv,
                 noise_model: NoiseModel = None,
                 seed: Optional[int] = None):
        super().__init__(env)
        self.noise_model = noise_model
        self.seed_val = seed
        self.rng = np.random.default_rng(self.seed_val)
        self.noise_model.rng = self.rng

    def reset(self, seed=None, options=None):
        """ Reset the environment and update seed for noise model."""
        result = self.env.reset(seed=seed, options=options)
        if seed is not None:
            self.seed_val = seed
            self.rng = np.random.default_rng(seed)
            self.noise_model.rng = self.rng
        return result

    def observe(self, agent):
        """Apply noise to the observation of other_agents."""
        raw_obs = self.env.observe(agent)
        observation_space = self.env.observation_space(agent)

        if isinstance(raw_obs, dict):
            noisy_obs = {}
            for sender, obs_values in raw_obs.items():
                if sender == agent:
                    noisy_obs[sender] = obs_values
                else:
                    if sender in observation_space:
                        noisy_obs[sender] = self.noise_model.apply(obs_values, observation_space[sender])
                    else:
                        noisy_obs[sender] = self.noise_model.apply(obs_values)
            return noisy_obs
        else:
            return self.noise_model.apply(raw_obs, observation_space)

    def last(self, observe: bool = True):
        """Applies noise to the last observation id observe is True."""
        obs, rew, term, trunc, info = self.env.last()
        if not observe:
            return None, rew, term, trunc, info

        current_agent = self.env.agent_selection
        observation_space = self.env.observation_space(self.env.agent_selection)

        if isinstance(raw_obs, dict):
            noisy_obs = {}
            for sender, obs_values in obs.items():
                if sender == current_agent:
                    noisy_obs[sender] = obs_values
                else:
                    if sender in observation_space:
                        noisy_obs[sender] = self.noise_model.apply(obs_values, observation_space[sender])
                    else:
                        noisy_obs[sender] = self.noise_model.apply(obs_values)
            return noisy_obs
        else:
            return self.noise_model.apply(obs, observation_space), rew, term, trunc, info

    def set_noise_model(self, model: NoiseModel):
        """Set custom noise model."""
        model.rng = self.rng
        self.noise_model = model
