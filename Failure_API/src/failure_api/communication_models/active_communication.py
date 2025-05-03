import numpy as np
from typing import Dict, Optional, Union, List, Tuple, Any, Callable


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
            matrix = np.ones((num_agents, num_agents), dtype=float)
            np.fill_diagonal(matrix, 0.0)  # Disable self-communication
            self.matrix = matrix

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

        self.matrix[sender_idx][receiver_idx] = bandwidth_value

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
        return (self.matrix > threshold).astype(bool)

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

    def reset(self):
        """Resets the communication matrix to the specified fill value (default = fully connected)."""
        self.matrix = np.ones((len(self.agent_ids), len(self.agent_ids)), dtype=float)

    def get_state(self) -> np.ndarray:
        """Gets a copy of the current state"""
        return self.matrix.copy()
