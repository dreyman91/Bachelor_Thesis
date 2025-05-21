import numpy as np
from typing import List, Optional
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
                 rng: Optional[np.random.Generator] = None
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
        self.rng = rng or np.random.default_rng()

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
        agent_ids = comms_matrix.agent_ids
        N = len(agent_ids)

        # Generate full NxN matrix of Bernoulli outcomes (True = success)
        success_mask = self.rng.random((N, N)) >= self.failure_prob

        # Remove self-connections
        np.fill_diagonal(success_mask, False)

        # Compute bandwidth matrix: 0.0 for failure, max_bandwidth for success
        bandwidth_matrix = np.where(success_mask, self.max_bandwidth, 0.0)

        comms_matrix.set_matrix(bandwidth_matrix)

        if self.failure_prob == 0.0:
            state = comms_matrix.get_state()
            assert np.all(state[~np.eye(len(state), dtype=bool)]), "[BUG] Failure at p=0.0 (excluding self-links)"

    @staticmethod
    def create_initial_matrix(agent_ids: List[str]) -> np.ndarray:
        """
        Create an initial connectivity matrix with all agents connected except to themselves.
        """
        n = len(agent_ids)
        matrix = np.ones((n, n), dtype=bool)
        np.fill_diagonal(matrix, False)  # No self-connections
        return matrix
