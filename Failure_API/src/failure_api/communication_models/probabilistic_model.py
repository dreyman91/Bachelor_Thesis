import numpy as np
from typing import List
from .active_communication import ActiveCommunication
from  .base_communication_model import CommunicationModels


class ProbabilisticModel(CommunicationModels):
    """
    ProbabilisticModel assigns random communication bandwidths.

    Each agent pair receives:
    - With probability `failure_prob`: bandwidth = 0.0
    - Else: bandwidth = max_bandwidth
    """
    def __init__(self,
                 agent_ids: List[str],
                 failure_prob: float,
                 max_bandwidth: float = 1.0,
                 ):
        self.agent_ids = agent_ids
        self.failure_prob = failure_prob
        self.max_bandwidth = max_bandwidth
        super().__init__()
    def update_connectivity(self, comms_matrix: ActiveCommunication):
        """
        Randomly disables communication between agents based on the failure probability.

        Args:
            comms_matrix: The connectivity matrix to update
        """
        for sender in comms_matrix.agent_ids:
            for receiver in comms_matrix.agent_ids:
                if sender != receiver:
                    if self.rng.random() < self.failure_prob:
                        bandwidth = 0.0
                    else:
                        bandwidth = self.max_bandwidth
                    comms_matrix.update(sender, receiver, bandwidth)
    @staticmethod
    def create_initial_matrix(agent_ids: List[str]) -> np.ndarray:
        """
        Initializes a fully connected matrix.
        """
        return np.ones((len(agent_ids), len(agent_ids)), dtype=bool)