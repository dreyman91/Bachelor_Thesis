import numpy as np
from typing import List, Optional, Callable
from abc import ABC, abstractmethod
from .active_communication import ActiveCommunication


class CommunicationModels(ABC):
    """Abstract base class for communication communication_models."""

    def __init__(self, seed: Optional[int] = 42, rng=None,):
        self.rng = rng if rng is not None else np.random.default_rng(seed)

    def set_rng(self, rng):
        """
        Set the shared RNG from the wrapper
        """
        self.rng = rng

    @abstractmethod
    def update_connectivity(self, comms_matrix: ActiveCommunication):
        """Updates the ActiveCommunication and inject failures"""
        pass

    @staticmethod
    @abstractmethod
    def create_initial_matrix(agent_ids: List[str]) -> np.ndarray:
        """Creates the initial ActiveCommunication matrix before failures"""
        pass
