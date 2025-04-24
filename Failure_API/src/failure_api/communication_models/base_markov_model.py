from typing import Callable, List, Dict
import numpy as np
from .active_communication import ActiveCommunication
from .base_communication_model import CommunicationModels
from collections import defaultdict


class BaseMarkovModel(CommunicationModels):
    """
        A Markov-based communication model where each communication link (sender -> receiver)
        is governed by a 2-state Markov chain (Connected <-> Disconnected).

        This implements a pure Markov chain with fixed transition probabilities.
        """

    def __init__(self,
                 agent_ids: List[str],
                 transition_probabilities: Dict[tuple[str, str], np.ndarray]
                 ):
        super().__init__()
        self.agent_ids = agent_ids
        self.transition_probabilities = transition_probabilities

        # State: Lazy initialization
        self.state = defaultdict(lambda: 1)

    def get_transition_matrix(self, sender: str, receiver: str) -> np.ndarray:
        """
        Gets the transition probability matrix for a specific link.
        Args:
            sender: Source agent ID
            receiver: Destination agent ID

        Returns:
            Transition probability matrix
        """
        key = (sender, receiver)
        return self.transition_probabilities.get(key, np.array([[0.9, 0.1],
                                                                [0.1, 0.9]]))

    def _update_pair(self, sender: str, receiver: str, comms_matrix: ActiveCommunication):
        """
        Updates the state for a single sender-receiver communication link.
        Args:
            sender: Source agent ID
            receiver: Destination agent ID
            comms_matrix: Communication matrix to update
        """
        current_state = self.state[(sender, receiver)]
        matrix = self.get_transition_matrix(sender, receiver)

        # Sample the next state from the transition matrix
        next_state = int(self.rng.choice([0, 1], p=matrix[current_state]))
        self.state[(sender, receiver)] = next_state
        comms_matrix.update(sender, receiver, next_state == 1)

    def update_connectivity(self, comms_matrix: ActiveCommunication):
        """
        Updates the connectivity status for all agent pairs.
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
