import numpy as np
from typing import Dict, Optional, Union, List, Tuple, Any, Callable
from abc import ABC, abstractmethod
import warnings
from gymnasium import spaces

class NoiseModel(ABC):
    """Base class for noise models."""

    def __init__(self, rng=None):
        self.rng = rng or np.random.default_rng()

    def set_rng(self, rng):
        """Set random number generator."""
        self.rng = rng

    @abstractmethod
    def apply(self, obs: Any):
        """Appy noise to an Observation."""
        pass


class GaussianNoise(NoiseModel):
    def __init__(self, mean: float = 0.0, std: float = 0.0, rng=None):
        super().__init__(rng)
        self.mean = mean
        self.std = std

    def apply(self, obs: Any, observation_space: Optional[spaces.Space] = None) -> Any:
        if isinstance(obs, np.ndarray):
            if observation_space is not None and isinstance(observation_space, spaces.Box):
                noise = self.rng.normal(loc=self.mean, scale=self.std, size=obs.shape)
                noisy_obs = obs + noise
                noisy_obs = np.clip(noisy_obs, observation_space.low, observation_space.high)
                return noisy_obs

            else: # if not known Box space, just add noise
                return obs + self.rng.normal(loc=self.mean, scale=self.std, size=obs.shape)

        elif isinstance(obs, dict):
            noisy_obs = {}
            for k, v in obs.items():
                if isinstance(v, np.ndarray):
                    if observation_space and k in observation_space and isinstance(observation_space[k], spaces.Box):
                        noise = self.rng.normal(loc=self.mean, scale=self.std, size=v.shape)
                        noise_arr = v + noise
                        noise_arr = np.clip(noise_arr, observation_space[k].low, observation_space[k].high)
                        noisy_obs[k] = noise_arr
                    else:  # No Observation space, or not a box
                        noisy_obs[k] = v + self.rng.normal(self.mean, self.std, size=v.shape)

                elif isinstance(v, (int, float, np.integer)):  # Handle Discrete and MultiDiscrete
                    if observation_space and k in observation_space:
                        if isinstance(observation_space[k], spaces.Discrete):
                            noise_val = v + self.rng.normal(self.mean, self.std)
                            clipped_val = np.clip(noise_val, 0, observation_space[k].n - 1)
                            noisy_obs[k] = int(round(clipped_val))
                        elif isinstance(observation_space[k], spaces.MultiDiscrete):
                            noise_val = v + self.rng.normal(self.mean, self.std)
                            clipped_val = np.clip(noise_val, 0, observation_space[k].nvec - 1)
                            noisy_obs[k] = np.round(clipped_val).astype(int)
                        elif isinstance(observation_space[k], spaces.Box):
                            noise = self.rng.normal(self.mean, self.std)
                            noise_arr = float(v) + noise
                            noise_arr = np.clip(noise_arr, observation_space[k].low, observation_space[k].high)
                            noisy_obs[k] = noise_arr
                        else:
                            noisy_obs[k] = v + self.rng.normal(self.mean, self.std)
                else:
                    noisy_obs[k] = v
            return noisy_obs
        else:
            return obs


class LaplacianNoise(NoiseModel):
    """Applies Laplacian Noise to the Observation"""
    def __init__(self, loc: float = 0.0, scale: float = 0.1, rng=None):
        super().__init__(rng)
        self.loc = loc
        self.scale = scale

    def apply(self, obs: Any, observation_space: Optional[spaces.Space] = None) -> Any:
        if isinstance(obs, np.ndarray):
            if observation_space is not None and isinstance(observation_space, spaces.Box):
                noise = self.rng.laplace(loc=self.loc, scale=self.scale, size=obs.shape)
                noisy_obs = obs + noise
                noisy_obs = np.clip(noisy_obs, observation_space.low, observation_space.high)
                return noisy_obs
            else:  # if not known Box space, just add noise
                return obs + self.rng.laplace(loc=self.loc, scale=self.scale, size=obs.shape)

        elif isinstance(obs, dict):
            noisy_obs = {}
            for k, v in obs.items():
                if isinstance(v, np.ndarray):
                    if observation_space and k in observation_space and isinstance(observation_space[k], spaces.Box):
                        noise = self.rng.laplace(loc=self.loc, scale=self.scale, size=v.shape)
                        noise_arr = v + noise
                        noise_arr = np.clip(noise_arr, observation_space[k].low, observation_space[k].high)
                        noisy_obs[k] = noise_arr
                    else:  # No Observation space, or not a box
                        noisy_obs[k] = v + self.rng.laplace(self.loc, self.scale, size=v.shape)

                elif isinstance(v, (int, float, np.integer)):  # Handle Discrete and MultiDiscrete
                    if observation_space and k in observation_space:
                        if isinstance(observation_space[k], spaces.Discrete):
                            noise_val = v + self.rng.laplace(self.loc, self.scale)
                            clipped_val = np.clip(noise_val, 0, observation_space[k].n - 1)
                            noisy_obs[k] = int(round(clipped_val))
                        elif isinstance(observation_space[k], spaces.MultiDiscrete):
                            noise_val = v + self.rng.laplace(self.loc, self.scale)
                            clipped_val = np.clip(noise_val, 0, observation_space[k].nvec - 1)
                            noisy_obs[k] = np.round(clipped_val).astype(int)
                        elif isinstance(observation_space[k], spaces.Box):
                            noise = self.rng.laplace(self.loc, self.scale)
                            noise_arr = float(v) + noise
                            noise_arr = np.clip(noise_arr, observation_space[k].low, observation_space[k].high)
                            noisy_obs[k] = noise_arr
                        else:
                            noisy_obs[k] = v + self.rng.laplace(self.loc, self.scale)
                else:
                    noisy_obs[k] = v
            return noisy_obs
        else:
            return obs


class CustomNoise(NoiseModel):
    def __init__(self, noise_fn: Callable[[Any, Optional[spaces.Space]], Any], rng=None):
        super().__init__(rng)
        self.noise_fn = noise_fn
        if not callable(noise_fn):
            raise ValueError("Noise function must be callable")

    def apply(self, obs: Any, observation_space: Optional[spaces.Space] = None) -> Any:
        noisy_obs = self.noise_fn(obs, observation_space)
        return noisy_obs