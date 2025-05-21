import numpy as np
from typing import Dict, Optional, Union, List, Tuple, Any, Callable
from scipy.sparse import  lil_matrix, csr_matrix


class ActiveCommunication:
    """Represents the actual communication state between agents as a bandwidth matrix.

    This model tracks bandwidth between agents as a floating-point values. A value > 0.0 means communication is possible
    and its magnitude can be used to represent link quality, throughput, or capacity.
    """

    def __init__(self,
                 agent_ids: List[str],
                 initial_matrix: Optional[np.ndarray] = None):
        self.agent_ids = agent_ids
        self.agent_id_to_index = {agent: i for i, agent in enumerate(self.agent_ids)}
        num_agents = len(agent_ids)

        if initial_matrix is not None:
            self._validate_matrix(initial_matrix, num_agents)
            self.matrix = initial_matrix.astype(bool)
        else:
            self.matrix = lil_matrix((num_agents, num_agents), dtype=float)
            self.matrix[:, :] = 1.0
            for i in range(num_agents):
                self.matrix[i, i] = 0.0

    @staticmethod
    def _validate_matrix(matrix, num_agents):
        if matrix.shape != (num_agents, num_agents):
            raise ValueError(f"Matrix must be {num_agents} x {num_agents}")
        if not np.all(np.diag(matrix) > 0.0):
            raise ValueError(f"Self-Communication bandwidth must always be positive")
        if not np.issubdtype(matrix.dtype, np.floating):
            raise ValueError("Matrix must be of float type for bandwidth representation")

    def update(self, sender: str, receiver: str, bandwidth: float):
        """
        Updates the bandwidth value between sender and receiver.
        A value of 0.0 disables communication; higher values increase communication quality.
        """
        sender_idx = self.agent_id_to_index[sender]
        receiver_idx = self.agent_id_to_index[receiver]

        # Ensure conversion from boolean to float
        if isinstance(bandwidth, bool):
            bandwidth_value = 1.0 if bandwidth else 0.0
        else:
            bandwidth_value = float(bandwidth)

        self.matrix[sender_idx, receiver_idx] = bandwidth_value

    def get_bandwidth(self, sender: str, receiver: str) -> float:
        """
        Returns the bandwidth value from sender to receiver
        """
        i = self.agent_id_to_index[sender]
        j = self.agent_id_to_index[receiver]
        return float(self.matrix[i, j])

    def can_communicate(self, sender: str, receiver: str, threshold: float = 0.0) -> bool:
        """
        Returns True if the bandwidth is above the given threshold (default is 0.0).
        This supports both binary and threshold boolean logic.
        """
        return self.get_bandwidth(sender, receiver) > threshold

    def get_boolean_matrix(self, threshold: float = 0.0) -> np.ndarray:
        """
        Returns a boolean matrix indicating connectivity for each pair using the given threshold.
        """
        if hasattr(self.matrix, "toarray"):
            dense_matrix = self.matrix.toarray()
        else:
            dense_matrix = self.matrix
        return (dense_matrix > threshold).astype(bool)

    def get_blocked_agents(self, agent: str) -> List[str]:
        """ Returns a list of agents that cannot communicate with the given agent under the thresholded logic."""
        agent_idx = self.agent_id_to_index[agent]
        blocked_indices = np.where(self.matrix[agent_idx] == False)[0]
        blocked_agents = [self.agent_ids[i] for i in blocked_indices if self.agent_ids[i] != agent]
        return blocked_agents

    def get(self, sender: str, receiver: str) -> bool:
        """
        Returns True if the bandwidth from sender to receiver is greater than 0.
        Acts like a boolean connectivity check.
        """
        i = self.agent_id_to_index[sender]
        j = self.agent_id_to_index[receiver]
        return self.matrix[i, j] > 0.0

    def as_numpy(self, threshold: float = 0.0) -> np.ndarray:
        """Returns a dense boolean matrix showing communication links"""
        return (self.matrix.toarray() > threshold).astype(int)

    def set_matrix(self, matrix: np.ndarray):
        self.matrix = matrix

    def reset(self, agent_ids: List[str]=None):
        """Resets the communication matrix to the specified fill value (default = fully connected)."""
        if agent_ids is not None:
            self.agent_ids = agent_ids
            self.agent_id_to_index = {agent: i for i, agent in enumerate(agent_ids)}

        num_agents = len(self.agent_ids)
        self.matrix = lil_matrix((num_agents, num_agents), dtype=float)
        self.matrix[:, :] = 1.0
        for i in range(num_agents):
            self.matrix[i, i] = 0.0

    def get_state(self) -> np.ndarray:
        """Gets a copy of the current state"""
        return self.matrix.copy()
