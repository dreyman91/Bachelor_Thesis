import numpy as np
from typing import List, Optional, Callable
from abc import ABC, abstractmethod
import warnings
from .active_communication import ActiveCommunication
from joblib import Parallel, delayed


class CommunicationModels(ABC):
    """Abstract base class for communication communication_models."""

    def __init__(self, seed: Optional[int] = 42, rng=None, parallel_executor: Optional[Callable] = None):
        self.rng = rng if rng is not None else np.random.default_rng(seed)
        self.parallel_executor = parallel_executor or Parallel(n_jobs=-1)

    def set_rng(self, rng):
        """
        Set the shared RNG from the wrapper
        """
        self.rng = rng

    def set_parallel_executor(self, parallel_executor):
        self.parallel_executor = parallel_executor

    @abstractmethod
    def update_connectivity(self, comms_matrix: ActiveCommunication):
        """Updates the ActiveCommunication and inject failures"""
        pass

    @staticmethod
    @abstractmethod
    def create_initial_matrix(agent_ids: List[str]) -> np.ndarray:
        """Creates the initial ActiveCommunication matrix before failures"""
        pass
