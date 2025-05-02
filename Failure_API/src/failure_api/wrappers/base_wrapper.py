import numpy as np
from pettingzoo.utils.wrappers import BaseWrapper
from typing import Optional


class BaseWrapper(BaseWrapper):
    """Shared functionality (e.g., seeding)."""

    def __init__(self, env, seed: Optional[int] = None):
        super().__init__(env)
        self.seed_val = seed
        self.rng = np.random.default_rng(seed)

    def reset_rng(self, seed: Optional[int] = None):
        """Update RNG and propagate to internal communication_models."""
        self.seed_val = seed
        self.rng = np.random.default_rng(seed)
