import numpy as np
from typing import Dict, Optional, Union, List, Tuple, Any, Callable
from abc import ABC, abstractmethod
import warnings
from .active_communication import ActiveCommunication

class CommunicationModels(ABC):
    """Abstract base class for communication models."""

    @abstractmethod
    def update_connectivity(self, comms_matrix: ActiveCommunication):
        """Updates the ActiveCommunication and inject failures"""
        pass

    @staticmethod
    @abstractmethod
    def create_initial_matrix(agent_ids: List[str]) -> np.ndarray:
        """Creates the initial ActiveCommunication matrix before failures"""
        pass


class DistanceModel(CommunicationModels):
    def __init__(self, agent_ids: List[str], distance_threshold: float, pos_fn: Callable[[], Dict[str, np.ndarray]],
                 failure_prob: float = 0.0, seed: Optional[int] = None):
        self.agent_ids = agent_ids
        self.distance_threshold = distance_threshold
        self.pos_fn = pos_fn
        self.failure_prob = failure_prob
        self.rng = np.random.default_rng(seed)

    def update_connectivity(self, comms_matrix: ActiveCommunication):
        positions = self.pos_fn()
        for sender in comms_matrix.agent_ids:
            for receiver in comms_matrix.agent_ids:
                if sender != receiver:
                    dist = np.linalg.norm(positions[sender] - positions[receiver])
                    can_communicate = dist <= self.distance_threshold

                    if can_communicate and self.rng.random() < self.failure_prob:
                        can_communicate = False
                    comms_matrix.update(sender, receiver, can_communicate)

    @staticmethod
    def create_initial_matrix(agent_ids: List[str]) -> np.ndarray:
        return np.ones((len(agent_ids), len(agent_ids)), dtype=bool)


class ProbabilisticModel(CommunicationModels):
    def __init__(self, agent_ids: List[str], failure_prob: float, seed: Optional[int] = None):
        self.agent_ids = agent_ids
        self.failure_prob = failure_prob
        self.rng = np.random.default_rng(seed)

    def update_connectivity(self, comms_matrix: ActiveCommunication):
        for sender in comms_matrix.agent_ids:
            for receiver in comms_matrix.agent_ids:
                if sender != receiver:
                    if self.rng.random() < self.failure_prob:
                        comms_matrix.update(sender, receiver, False)
    @staticmethod
    def create_initial_matrix(agent_ids: List[str]) -> np.ndarray:
        return np.ones((len(agent_ids), len(agent_ids)), dtype=bool)


class MatrixModel(CommunicationModels):
    def __init__(self, agent_ids: List[str], comms_matrix_values: np.ndarray, failure_prob: float = 0.0,
                 seed: Optional[int] = None):
        self.agent_ids = agent_ids
        self.comms_matrix_values = comms_matrix_values
        self.failure_prob = failure_prob
        self.rng = np.random.default_rng(seed)

    def update_connectivity(self, comms_matrix: ActiveCommunication):
        for sender in comms_matrix.agent_ids:
            for receiver in comms_matrix.agent_ids:
                sender_idx = comms_matrix.agent_id_to_index[sender]
                receiver_idx = comms_matrix.agent_id_to_index[receiver]
                allowed = self.comms_matrix_values[sender_idx, receiver_idx]
                if allowed and self.rng.random() < self.failure_prob:
                    allowed = False
                comms_matrix.update(sender, receiver, allowed)

    @staticmethod
    def create_initial_matrix(agent_ids: List[str]) -> np.ndarray:
        return np.array([], dtype=bool)


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
