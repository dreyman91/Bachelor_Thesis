import numpy as np
from typing import List
from .active_communication import ActiveCommunication
from .base_communication_model import CommunicationModels

class MatrixModel(CommunicationModels):
    """
    MatrixModel uses a user-defined bandwidth matrix for connectivity.
    """
    def __init__(self,
                 agent_ids: List[str],
                 comms_matrix_values: np.ndarray,
                 failure_prob: float = 0.0
                 ):
        self.agent_ids = agent_ids
        self.comms_matrix_values = comms_matrix_values.astype(float)
        self.failure_prob = failure_prob
        super().__init__()

    def update_connectivity(self, comms_matrix: ActiveCommunication):
        """
        Applies failure probability on top of predefined matrix and updates comms_matrix accordingly.
        """
        for sender in comms_matrix.agent_ids:
            for receiver in comms_matrix.agent_ids:
                sender_idx = comms_matrix.agent_id_to_index[sender]
                receiver_idx = comms_matrix.agent_id_to_index[receiver]
                bandwidth = self.comms_matrix_values[sender_idx, receiver_idx]
                if bandwidth  > 0.0 and self.rng.random() < self.failure_prob:
                    bandwidth = 0.0
                comms_matrix.update(sender, receiver, bandwidth)
    @staticmethod
    def create_initial_matrix(agent_ids: List[str]) -> np.ndarray:
        """
        Empty placeholder matrix — user is expected to supply a custom one.
        """
        return np.array([], dtype=bool)