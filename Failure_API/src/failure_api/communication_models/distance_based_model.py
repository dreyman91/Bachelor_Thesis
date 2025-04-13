import numpy as np
from typing import List, Dict, Callable, Any
from .active_communication import ActiveCommunication
from  .base_communication_model import CommunicationModels

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
        super().__init__()
        self.agent_ids = agent_ids
        self.distance_threshold = distance_threshold
        self.pos_fn = pos_fn
        self.failure_prob = failure_prob
        self.max_bandwidth = max_bandwidth
    def update_connectivity(self, comms_matrix: ActiveCommunication):
        """
        Update which agents can communicate based on distance and failure probability.
        """
        positions = self.pos_fn()
        for sender in comms_matrix.agent_ids:
            for receiver in comms_matrix.agent_ids:
                if sender != receiver:
                    dist = np.linalg.norm(positions[sender] - positions[receiver])
                    if dist <= self.distance_threshold:
                        bandwidth = self.max_bandwidth * (1 - (dist / self.distance_threshold))
                        if self.rng.random() < self.failure_prob:
                            bandwidth = 0.0
                    else:
                        bandwidth = 0.0

                    comms_matrix.update(sender, receiver, bandwidth)

    @staticmethod
    def create_initial_matrix(agent_ids: List[str]) -> np.ndarray:
        """
        Initializes a fully connected matrix.
        """
        return np.ones((len(agent_ids), len(agent_ids)), dtype=bool)