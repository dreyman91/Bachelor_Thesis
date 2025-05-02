import numpy as np
from typing import List
from .active_communication import ActiveCommunication
from .base_communication_model import CommunicationModels


class ProbabilisticModel(CommunicationModels):
    """
    A model that assigns communication bandwidth based on independent Bernoulli trials.

    For each sender-receiver pair and each time step, this model:
    - With probability p: Sets bandwidth to 0.0 (communication failure)
    - With probability (1-p): Sets bandwidth to max_bandwidth (successful communication)

    This implements a memoryless stochastic process where communication
    successes/failures are independent across links and time steps.
    """

    def __init__(self,
                 agent_ids: List[str],
                 failure_prob: float,
                 max_bandwidth: float = 1.0,
                 ):
        super().__init__()
        self.agent_ids = agent_ids

        # Parameter validation
        if not 0 <= failure_prob <= 1:
            raise ValueError("Failure probability must be between 0 and 1")
        if max_bandwidth < 0:
            raise ValueError("Max Bandwidth must be positive")

        self.failure_prob = failure_prob
        self.max_bandwidth = max_bandwidth

    def get_failure_probability(self, sender: str, receiver: str) -> float:
        """
        Get the failure probability for a specific sender-receiver pair.
        """
        return self.failure_prob

    def update_connectivity(self, comms_matrix: ActiveCommunication):
        """
        Update connectivity matrix using independent Bernoulli trials.

        For each sender-receiver pair, randomly determine if communication
        succeeds or fails based on the failure probability.
        """

        for sender in sorted(comms_matrix.agent_ids):
            for receiver in sorted(comms_matrix.agent_ids):
                if sender == receiver:
                    continue
                # Get failure probability for this specific link
                p_fail = self.get_failure_probability(sender, receiver)

                # Perform Bernoulli trial
                if self.rng.random() < p_fail:
                    bandwidth = 0.0  # Failed
                else:
                    bandwidth = self.max_bandwidth  # Success

                # Update matrix
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
