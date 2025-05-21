from typing import Callable, List, Dict, Optional
import numpy as np
from .active_communication import ActiveCommunication
from .base_communication_model import CommunicationModels
from collections import defaultdict


class BaseMarkovModel(CommunicationModels):
    """
         A pure Markov chain model for communication reliability.

        This model treats each communication link (sender -> receiver) as governed
        by a 2-state Markov chain representing connectivity:
        - State 0: Disconnected
        - State 1: Connected

        The Markov chain transitions are defined by 2×2 matrices where:
        - P[0,0]: Probability of staying disconnected
        - P[0,1]: Probability of changing from disconnected to connected
        - P[1,0]: Probability of changing from connected to disconnected
        - P[1,1]: Probability of staying connected

        Each link can have its own transition matrix, or a default is provided.
    """

    def __init__(self,
                 agent_ids: List[str],
                 transition_probabilities: Optional[Dict[tuple[str, str], np.ndarray]] = None,
                 default_matrix: Optional[np.ndarray] = None
                 ):
        """
        Initialize the BaseMarkovModel.

        Args:
            agent_ids: List of agent names or IDs.
            transition_probabilities: Optional dict mapping (sender, receiver) pairs to 2x2 transition matrices.
            default_matrix: Optional fallback matrix used if a link is unspecified.
        """
        super().__init__()
        self.agent_ids = agent_ids
        self.transition_probabilities = transition_probabilities if transition_probabilities is not None else {}
        self.default_matrix = default_matrix if default_matrix is not None else np.array([
            [0.9, 0.1],  # From disconnected: 90% stay disconnected, 10% become connected
            [0.1, 0.9]  # From connected: 10% become disconnected, 90% stay connected
        ])

        if self.default_matrix.shape != (2, 2):
            raise ValueError("default_matrix must be 2x2")
        if not np.allclose(self.default_matrix.sum(axis=1), 1.0, atol=1e-6):
            raise ValueError("Each row of default_matrix must sum to 1.")

        # Validate transition matrices
        self._validate_transition_matrices()

        # State: Lazy initialization
        self.state = defaultdict(lambda: 1)

    def _validate_transition_matrices(self):
        """
        Validate that all transition matrices are properly formatted.

        A valid transition matrix must:
        1. Be a 2×2 numpy array
        2. Have rows that sum to 1.0 (or very close due to floating point)
        3. Have values between 0 and 1

        Raises warnings for invalid matrices and fixes them when possible.
        """

        for key, matrix in self.transition_probabilities.items():
            # Check shape
            if not isinstance(matrix, np.ndarray) or matrix.shape != (2, 2):
                logger.warning(f"Warning! Invalid transition matrix for {key}. Using default")
                self.transition_probabilities[key] = self.default_matrix.copy()
                continue

            # Check value range
            if np.any(matrix < 0) or np.any(matrix > 1):
                logger.warning(f"Warning! Transition matrix for {key} has invalid probability values. Expected [0, "
                               f"1]. Clipping")
                self.transition_probabilities[key] = np.clip(matrix, 0, 1)

            # Check row sums
            row_sums = matrix.sum(axis=1)
            if not np.allclose(row_sums, 1.0, rtol=1e-5):
                logger.warning(
                    f"Warning! Invalid probability values for  {key}. Row sums must be 1.0. Got: {row_sums}. "
                    f"Normalizing")
                self.transition_probabilities[key] = matrix / row_sums[:, np.newaxis]

    def get_transition_matrix(self, sender: str, receiver: str) -> np.ndarray:
        """
        Gets the transition probability matrix for a specific link.

        Returns: 2×2 transition probability matrix
        """
        key = (sender, receiver)
        return self.transition_probabilities.get(key, self.default_matrix)

    def _update_pair(self, sender: str, receiver: str, comms_matrix: ActiveCommunication):
        """
        Updates the state for a single sender-receiver communication link.
        Args:
            sender: Source agent ID
            receiver: Destination agent ID
            comms_matrix: Communication matrix to update
        """
        # Get current state
        current_state = self.state[(sender, receiver)]

        # get the transition matrix for this link
        matrix = self.get_transition_matrix(sender, receiver)

        # Sample the next state from the transition matrix
        next_state = int(self.rng.choice([0, 1], p=matrix[current_state]))

        # Update state and communication matrix
        self.state[(sender, receiver)] = next_state
        comms_matrix.update(sender, receiver, next_state == 1)

    def update_connectivity(self, comms_matrix: ActiveCommunication):
        """
        Update connectivity for all agent pairs based on Markov transitions.
        Args:
            comms_matrix: Communication matrix to update
            
        """
        for sender in self.agent_ids:
            for receiver in self.agent_ids:
                if sender != receiver:
                    self._update_pair(sender, receiver, comms_matrix)

    @staticmethod
    def create_initial_matrix(agent_ids: List[str]) -> np.ndarray:
        """ Returns a matrix with maximum potential connectivity set to True
        except for self-loops which are set to False"""
        n_agents = len(agent_ids)
        matrix = np.ones((n_agents, n_agents), dtype=bool)
        np.fill_diagonal(matrix, False)
        return matrix
