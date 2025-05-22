import numpy as np
from pettingzoo import AECEnv
from typing import Optional
from .base_wrapper import BaseWrapper
from .sharedobs_wrapper import SharedObsWrapper
from ..noise_models.base_noise_model import NoiseModel


class NoiseWrapper(BaseWrapper):
    def __init__(self,
                 env: AECEnv,
                 noise_model: NoiseModel = None,
                 seed: Optional[int] = None):
        wrapped_env = SharedObsWrapper(env)
        super().__init__(env)
        self.env = wrapped_env

        self.noise_model = noise_model
        self.reset_rng(seed)
        if self.noise_model:
            self.noise_model.set_rng(self.rng)

    def reset(self, seed=None, options=None):
        """ Reset the environment and update seed for a noise model."""
        result = self.env.reset(seed=seed, options=options)
        if seed is not None:
            self.reset_rng(seed)
            if self.noise_model:
                self.noise_model.set_rng(self.rng)
        return result

    def observe(self, agent):
        """Apply noise only to non-zero values in the observation."""
        raw_obs = self.env.observe(agent)
        if raw_obs is None:
            space = self.observation_space(agent)
            return {a: np.zeros_like(space[a].low) for a in space} if isinstance(space, dict) else np.zeros_like(
                space.low)

        raw_obs = raw_obs.copy()
        observation_space = self.env.observation_space(agent)

        # Handle dictionary-style observation (multi-agent)
        if isinstance(raw_obs, dict):
            noisy_obs = {}
            for sender, obs_values in raw_obs.items():
                if sender == agent:
                    noisy_obs[sender] = obs_values
                elif isinstance(obs_values, np.ndarray) and obs_values.size > 0:
                    obs_values = obs_values.copy()

                    if sender in observation_space:
                        noisy = self.noise_model.apply(obs_values, observation_space[sender])
                    else:
                        noisy = self.noise_model.apply(obs_values)

                    # Preserve zero-masked values
                    result = np.where(obs_values != 0, noisy, obs_values)

                    # Debug: print noise norm on masked entries
                    zeroed = obs_values == 0
                    if np.any(zeroed):
                        print(f"[Debug] {agent}<-{sender} masked ||noise|| = {np.linalg.norm(result[zeroed])}")

                    noisy_obs[sender] = result
                else:
                    # Edge case: not a valid array or empty
                    noisy_obs[sender] = obs_values
            return noisy_obs
        # Handle flat single-agent observations
        elif isinstance(raw_obs, np.ndarray):
            noisy = self.noise_model.apply(raw_obs, observation_space)
            result = np.where(raw_obs != 0, noisy, raw_obs)

            zeroed = raw_obs == 0
            if np.any(zeroed):
                print("Mean magnitude of masked values after noise:", np.linalg.norm(result[zeroed]))

            return result
        return raw_obs

    def last(self, observe: bool = True):
        """Applies noise to the last observation id observe is True."""
        agent = self.env.agent_selection
        if agent not in self.env.agents:
            return None, 0.0, True, False, {}
        obs, rew, term, trunc, info = self.env.last()
        if not observe:
            return None, rew, term, trunc, info

        current_agent = self.env.agent_selection
        observation_space = self.env.observation_space(self.env.agent_selection)
        if obs is None:
            return None, rew, term, trunc, info

        if isinstance(obs, dict):
            noisy_obs = {}
            for sender, obs_values in obs.items():
                if sender == current_agent:
                    noisy_obs[sender] = obs_values
                else:
                    # create a copy of the original values
                    result = obs_values.copy()
                    # find non-zero values
                    non_zero_mask = obs_values != 0

                    if non_zero_mask.any():  # only apply noise if there are non-zero values
                        if sender in observation_space:
                            noisy_values = self.noise_model.apply(obs_values, observation_space[sender])
                        else:
                            noisy_values = self.noise_model.apply(obs_values)

                        # Extract noise component
                        noise = noisy_values - obs_values

                        # Apply noise to non-zero values
                        result[non_zero_mask] += noise[non_zero_mask]
                    noisy_obs[sender] = result
            return noisy_obs, rew, term, trunc, info
        else:
            #Flat observations
            non_zero_mask = obs != 0
            result = obs.copy()

            if non_zero_mask.any():
                noisy_obs = self.noise_model.apply(obs, observation_space)
                noise = noisy_obs - obs
                result[non_zero_mask] += noise[non_zero_mask]
            return result, rew, term, trunc, info

    def set_noise_model(self, model: NoiseModel):
        """Set custom noise model."""
        model.rng = self.rng
        self.noise_model = model