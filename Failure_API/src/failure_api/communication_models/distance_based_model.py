import numpy as np
from typing import List, Dict, Callable, Any
from .active_communication import ActiveCommunication
from .base_communication_model import CommunicationModels
import logging

logger = logging.getLogger(__name__)


class DistanceModel(CommunicationModels):
    """
    A communication model where agents can only communicate if they are within a certain distance.
    """

    def __init__(self,
                 agent_ids: List[str],
                 distance_threshold: float,
                 pos_fn: Callable[[], Dict[str, np.ndarray]],
                 failure_prob: float = 0.0,
                 max_bandwidth: float = 1.0
                 ):
        """
        Initialize the distance-based communication model.

        Parameters:
        agent_ids : List of unique identifiers for all agents
        position_fn : Function that returns current positions of all agents
        distance_threshold : Maximum distance at which communication is possible
        max_bandwidth : Maximum bandwidth when agents are at the same position
        failure_probability : Probability of random communication failure regardless of distance
        """
        super().__init__()

        # Parameter validation
        if distance_threshold <= 0.0:
            logger.warning("Distance threshold must be greater than zero")
        if not 0 <= failure_prob <= 1.0:
            logger.warning("Failure probability must be between 0 and 1")
        if max_bandwidth <= 0.0:
            logger.warning("Bandwidth must be greater than zero")

        self.agent_ids = agent_ids
        self.distance_threshold = distance_threshold
        self.pos_fn = pos_fn
        self.failure_prob = failure_prob
        self.max_bandwidth = max_bandwidth

    def calculate_distance(self, pos1: np.ndarray, pos2: np.ndarray) -> float:
        """
        Calculate the Euclidean distance between two positions.

        Parameters:
        pos1 : First position vector
        pos2 : Second position vector

        Returns: Euclidean distance between positions
        """
        return float(np.linalg.norm(pos1 - pos2))

    def calculate_bandwidth(self, distance: float) -> float:
        """
        Calculate bandwidth based on distance between agents.

        In the base model, bandwidth decreases linearly with distance:
        - At distance 0: bandwidth = max_bandwidth
        - At distance = threshold: bandwidth = 0
        - Beyond threshold: bandwidth = 0
        :param distance: Distance between agents
        :return calculated bandwidth
        """

        if distance > self.distance_threshold:
            return 0.0

        # Linear degradation with distance
        return self.max_bandwidth * (1.0 - (distance / self.distance_threshold))

    def apply_random_failure(self, bandwidth: float) -> float:
        """
        Apply random failure probability to bandwidth.

        :param bandwidth: Base bandwidth before random failure
        :return: Final bandwidth after potential random failure
        """
        if bandwidth > 0 and self.rng.random() < self.failure_prob:
            return 0.0
        return bandwidth

    def update_connectivity(self, comms_matrix: ActiveCommunication):
        """
        Update connectivity matrix based on agent distances and random failures.
        """
        # Get current positions of all agents
        positions = self.pos_fn()

        # Update connectivity for each agent pair
        for sender in self.agent_ids:
            sender_pos = positions.get(sender)
            if sender_pos is None:
                continue

            for receiver in self.agent_ids:
                if sender == receiver:
                    continue

                receiver_pos = positions.get(receiver)
                if receiver_pos is None:
                    continue

                # Calculate distance
                distance = self.calculate_distance(sender_pos, receiver_pos)

                # Calculate distance-based bandwidth
                bandwidth = self.calculate_bandwidth(distance)

                # Apply random failure
                bandwidth = self.apply_random_failure(bandwidth)

                comms_matrix.update(sender, receiver, bandwidth)

    @staticmethod
    def create_initial_matrix(agent_ids: List[str]) -> np.ndarray:
        """
        Create an initial connectivity matrix with all agents connected except to themselves.
        """
        n = len(agent_ids)
        matrix = np.ones((n, n), dtype=bool)
        np.fill_diagonal(matrix, False)  # No self-connections
        return matrix
