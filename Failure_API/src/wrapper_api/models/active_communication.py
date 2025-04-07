import numpy as np
from typing import Dict, Optional, Union, List, Tuple, Any, Callable

class ActiveCommunication:
    """Represents the actual communication state between agents."""

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
            self.matrix = np.ones((num_agents, num_agents), dtype=bool)

    @staticmethod
    def _validate_matrix(matrix, num_agents):
        if matrix.shape != (num_agents, num_agents):
            raise ValueError(f"Matrix must be {num_agents} x {num_agents}")
        if not np.all(np.diag(matrix) == True):
            raise ValueError(f"Self Communication must always be enabled")
        if not np.all((matrix >= False) & (matrix <= True)):
            raise ValueError(f"Matrix must be either False or True")

    def can_communicate(self, sender: str, receiver: str) -> bool:
        """Returns True if the agent can communicate with other agents."""
        sender_idx = self.agent_id_to_index[sender]
        receiver_idx = self.agent_id_to_index[receiver]
        return bool(self.matrix[sender_idx, receiver_idx])

    def update(self, sender: str, receiver: str, can_communicate: bool):
        """Updates communication status"""
        sender_idx = self.agent_id_to_index[sender]
        receiver_idx = self.agent_id_to_index[receiver]
        self.matrix[sender_idx, receiver_idx] = can_communicate

    def get_blocked_agents(self, agent: str) -> List[str]:
        """Returns a list of agents that cannot communicate with the given agent."""
        agent_idx = self.agent_id_to_index[agent]
        blocked_indices = np.where(self.matrix[agent_idx] == False)[0]
        blocked_agents = [self.agent_ids[i] for i in blocked_indices if self.agent_ids[i] != agent]
        return blocked_agents

    def reset(self):
        """Resets the communication matrix to fully connected"""
        self.matrix = np.ones((len(self.agent_ids), len(self.agent_ids)), dtype=bool)

    def get_state(self) -> np.ndarray:
        """Gets a copy of the current state"""
        return self.matrix.copy()