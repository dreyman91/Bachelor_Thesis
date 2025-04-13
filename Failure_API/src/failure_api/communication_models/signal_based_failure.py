from .active_communication import ActiveCommunication
from .base_communication_model import CommunicationModels
from typing import List, Callable, Dict, Tuple, Optional
import numpy as np
from scipy.spatial import cKDTree



class SignalBasedModel(CommunicationModels):
    """
    A communication model that simulates realistic wireless signal propagation.

    This model determines connectivity between agents based on:
    1. Physical distance between agents (using inverse-square law for signal attenuation)
    2. Probabilistic packet loss that increases with distance
    3. Configurable transmission power and minimum signal thresholds

    Optimized for real-time applications using spatial indexing (KD-Tree) to efficiently
    compute agent proximity relationships.
    """

    def __init__(self,
                 agent_ids: List[str],
                 pos_fn: Callable[[], Dict[str, np.ndarray]],
                 tx_power: float = 15.0,
                 min_strength: float = 0.01,
                 dropout_alpha: float = 0.2,
                 ):
        """
        Initialize the signal-based communication model.

        Args:
            agent_ids: List of unique identifiers for all agents in the simulation
            position_fn: Callable that returns a dictionary mapping agent IDs to their position vectors
            tx_power: Transmission power of agents (higher values increase effective communication range)
            min_strength: Minimum signal strength required for communication to be possible
            dropout_alpha: Controls how quickly probability of packet loss increases with distance
                           (higher values = more aggressive dropout with distance)
        """
        if not isinstance(agent_ids, list) or not all(isinstance(a, str) for a in agent_ids):
            raise TypeError("agent_ids must be a list of strings.")

        if not callable(pos_fn):
            raise TypeError("pos_fn must be a callable.")

        if not isinstance(tx_power, (float, int)):
            raise TypeError("tx_power must be a float.")

        if not isinstance(min_strength, (float, int)):
            raise TypeError("min_strength must be a float.")

        if not isinstance(dropout_alpha, (float, int)):
            raise TypeError("dropout_alpha must be a float.")

        super().__init__()
        self.agent_ids = agent_ids
        self.pos_fn = pos_fn
        self.tx_power = tx_power
        self.min_strength = min_strength
        self.dropout_alpha = dropout_alpha
        self.id_to_idx = {aid: i for i, aid in enumerate(agent_ids)}

    def _signal_strength(self, dist: float)-> float:
        """
        Calculate signal strength between two agents based on distance.

        Uses the inverse-square law with a small softening term to prevent division by zero.
        Signal strength = transmit_power / (distance² + epsilon)

        :param dist: Euclidean distance between sender and receiver
        :return: The calculated signal strength (higher = stronger connection)
        """

        return self.tx_power / (dist**2 + 1e-6)

    def update_connectivity(self, comms_matrix: ActiveCommunication):
        """
                Update the connectivity matrix based on current agent positions and signal parameters.

                This method:
                1. Gets current positions of all agents
                2. Builds a KD-Tree for efficient spatial querying
                3. For each sender-receiver pair:
                   a. Calculates signal strength based on distance
                   b. Determines if signal strength exceeds minimum threshold
                   c. Applies probabilistic packet loss (more likely at greater distances)
                   d. Updates the connectivity matrix accordingly

                Args:
                    comms_matrix: The connectivity matrix to update
                """
        positions = self.pos_fn()
        coords = np.array([positions[aid] for aid in self.agent_ids])

        # guard clause
        if  coords.size == 0:
            return
        tree = cKDTree(coords)

        for i, sender in enumerate(self.agent_ids):
            sender_pos = positions[sender]
            _, neighbors = tree.query(sender_pos, k=len(self.agent_ids))
            if isinstance(neighbors, (int, np.integer)):
                neighbors = [neighbors]
            else:
                neighbors = neighbors.tolist()

            for j in neighbors:
                receiver = self.agent_ids[j]
                if sender == receiver:
                    continue

                dist = float(np.linalg.norm(positions[sender] - positions[receiver]))
                strength = self._signal_strength(dist)

                if strength < self.min_strength:
                    comms_matrix.update(sender, receiver, False)
                    continue

                p_success = np.exp(-self.dropout_alpha * dist)
                success = self.rng.random() < p_success

                comms_matrix.update(sender, receiver, success)

    @staticmethod
    def create_initial_matrix(agent_ids: List[str]) -> np.ndarray:
        n = len(agent_ids)
        matrix = np.ones((n, n))
        np.fill_diagonal(matrix, 0.0)
        return matrix