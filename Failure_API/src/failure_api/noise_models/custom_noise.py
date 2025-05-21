import numpy as np
from typing import Optional, Any, Callable
from gymnasium import spaces
from .base_noise_model import NoiseModel

class CustomNoise(NoiseModel):
    """User-defined function to apply custom noise to observations. """
    def __init__(self, noise_fn: Callable[[Any, Optional[spaces.Space]], Any]):
        super().__init__()
        self.noise_fn = noise_fn
        if not callable(noise_fn):
            raise ValueError("Noise function must be callable")

    def apply(self, obs: Any, observation_space: Optional[spaces.Space] = None) -> Any:
        noisy_obs = self.noise_fn(obs, observation_space)
        return noisy_obs