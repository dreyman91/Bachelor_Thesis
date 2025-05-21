from .active_communication import ActiveCommunication
from .base_communication_model import CommunicationModels
from typing import List, Callable, Dict, Tuple, Optional
import numpy as np
from scipy.spatial import cKDTree


class SignalBasedModel(CommunicationModels):
    """
     A physics-based model for wireless signal propagation between agents.

    This model simulates realistic wireless communication by:
    1. Calculating signal strength based on physical distance using inverse-square law
    2. Applying signal strength thresholds for minimal connectivity
    3. Modeling probabilistic packet loss that increases with distance
    4. Using spatial indexing for computational efficiency

    The model is based on fundamental principles of electromagnetic wave propagation,
    making it suitable for research on wireless networks, robot swarms, sensor networks,
    and other systems with physical communication constraints.
    """

    def __init__(self,
                 agent_ids: List[str],
                 pos_fn: Callable[[], Dict[str, np.ndarray]],
                 tx_power: float = 15.0,
                 min_strength: float = 0.01,
                 dropout_alpha: float = 0.2,
                 rng: Optional[np.random.Generator] = None
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

        super().__init__()

        if not isinstance(agent_ids, list) or not all(isinstance(a, str) for a in agent_ids):
            raise TypeError("agent_ids must be a list of strings.")

        if not callable(pos_fn):
            raise TypeError("pos_fn must be a callable.")

        if not isinstance(tx_power, (float, int)) or tx_power <= 0:
            raise TypeError("tx_power must be a positive number.")

        if not isinstance(min_strength, (float, int)) or min_strength < 0:
            raise TypeError("min_strength must be a non-negative number.")

        if not isinstance(dropout_alpha, (float, int)) or dropout_alpha < 0:
            raise TypeError("dropout_alpha must be a non-negative number.")

        self.agent_ids = agent_ids
        self.pos_fn = pos_fn
        self.tx_power = tx_power
        self.min_strength = min_strength
        self.dropout_alpha = dropout_alpha
        self.verbose = False
        self.rng = rng or np.random.default_rng()

        # Creating mapping from agent IDs to indices
        self.id_to_idx = {aid: i for i, aid in enumerate(agent_ids)}

    def calculate_signal_strength(self, sender_pos: np.ndarray, receiver_pos: np.ndarray) -> float:
        """
        Calculate signal strength between two agents based on their positions.

        Implements the inverse-square law of electromagnetic radiation:
        signal_strength = tx_power / (distance² + ε)

        Where ε is a small constant to prevent division by zero when
        agents are at the exact same position.
        """
        # Calculate euclidean distance between the agents
        distance = float(np.linalg.norm(sender_pos - receiver_pos))

        # Apply inverse square law with small epsilon to prevent division by zero
        return self.tx_power / (distance ** 2 + 1e-6)

    def calculate_packet_success_probability(self, distance: float) -> float:
        """
        Calculate the probability of successful packet transmission based on distance.

        Uses an exponential decay model: p_success = exp(-α * distance)

        Parameters:
        distance : Distance between sender and receiver

        Returns: Probability of successful packet transmission (0.0 to 1.0)
        """
        return np.exp(-self.dropout_alpha * distance)

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

        # Get current positions of all agents
        positions = self.pos_fn()

        # Get valid positions
        valid_positions = {aid: positions[aid] for aid in self.agent_ids if aid in positions}
        if not valid_positions:
            # Set all off-diagonal links to 0.0 (disconnected) if no valid positions
            for sender in self.agent_ids:
                for receiver in self.agent_ids:
                    if sender != receiver:
                        comms_matrix.update(sender, receiver, 0.0)
            return

        #  Extract position coordinates as a numpy array for KD-Tree
        coords = np.array([positions[aid] for aid in self.agent_ids])

        # guard clause if no valid positions
        if coords.size == 0:
            return

        if self.verbose:
            print(f"Coords array for KDTree: {coords.shape}")

        # Build KD-Tree for efficient spatial querie
        tree = cKDTree(coords)

        # Update connectivity for each pair
        for i, sender in enumerate(self.agent_ids):

            sender_pos = positions.get(sender)
            if sender_pos is None:
                # If sender has no position, mark all its outgoing connections as failed
                for receiver in self.agent_ids:
                    if receiver != sender:
                        comms_matrix.update (sender, receiver, 0.0)
                continue # Skip to next sender

            # find all neighbors
            _, neighbors = tree.query(sender_pos, k=len(self.agent_ids))

            # Convert to list if only one neighbor
            if isinstance(neighbors, (int, np.integer)):
                neighbors = [neighbors]
            else:
                neighbors = neighbors.tolist()

            # Process each neighbor
            for j in neighbors:
                receiver = self.agent_ids[j]

                # Skip self-connections
                if sender == receiver:
                    continue

                # Get receiver position
                receiver_pos = positions.get(receiver)
                if receiver_pos is None:
                    continue

                # Calculate distance between agents
                distance = float(np.linalg.norm(sender_pos - receiver_pos))

                # Calculate signal strength
                strength = self.calculate_signal_strength(sender_pos, receiver_pos)

                # Check if signal meets a minimum threshold
                if strength < self.min_strength - 1e-8:
                    comms_matrix.update(sender, receiver, False)
                    continue

                # Calculate the probability of successful transmission
                p_success = self.calculate_packet_success_probability(distance)

                # Determine if transmission succeeds
                success = self.rng.random() < p_success

                # Update connectivity matrix
                comms_matrix.update(sender, receiver, success)

    @staticmethod
    def create_initial_matrix(agent_ids: List[str]) -> np.ndarray:
        """ Create an initial connectivity matrix with all agents connected except to themselves."""
        n = len(agent_ids)
        matrix = np.ones((n, n))
        np.fill_diagonal(matrix, 0.0)
        return matrix
