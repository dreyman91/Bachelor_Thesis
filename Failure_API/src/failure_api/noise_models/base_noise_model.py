import numpy as np
from typing import Optional, Any
from abc import ABC, abstractmethod


class NoiseModel(ABC):
    """Abstract base class for noise generators in MARL environments."""

    def __init__(self):
        self.rng = None

    def set_rng(self, rng):
        """Set random number generator."""
        self.rng = rng

    @abstractmethod
    def apply(self, obs: Any):
        """Appy noise to an Observation."""
        pass





