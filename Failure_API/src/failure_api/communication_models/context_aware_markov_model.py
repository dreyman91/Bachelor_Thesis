from typing import Callable, List, Dict, Optional
import numpy as np
from .active_communication import ActiveCommunication
from .base_markov_model import BaseMarkovModel


class ContextAwareMarkovModel(BaseMarkovModel):
    """
        A context-aware extension of the base Markov model that allows for
        transition probabilities to be modified based on external factors.

        This model provides hooks for modifiers from the markov_chain_scenarios package
        to adjust the base transition probabilities.
        """

    def __init__(self,
                 agent_ids: List[str],
                 transition_probabilities: Dict[tuple[str, str], np.ndarray],
                 modifier_functions: Optional[List[Callable[[np.ndarray, str, str], np.ndarray]]] = None
                 ):
        super().__init__()
        self.agent_ids = agent_ids
        self.transition_probabilities = transition_probabilities
        self.modifier_functions = modifier_functions

    def _adjust_matrix(self, matrix: np.ndarray, sender: str, receiver: str) -> np.ndarray:

        # Get the base matrix
        matrix = super().get_transition_matrix(sender, receiver).copy()

        # Apply each modifier function in sequence
        for modifier_fn in self.modifier_functions:
            matrix = modifier__fn(matrix, sender, receiver)

        matrix = np.clip(matrix, 0.0, 1.0)
        row_sums = matrix.sum(axis=1, keepdims=True)

        # Avoid division by matrix
        row_sums = np.where(row_sums == 0.0, 1.0, row_sums)
        matrix = matrix / row_sums

        return matrix

        # Context effects
        if self.context_fn:
            ctx = self.context_fn()


        # Time dependent behavior
        if self.time_fn:
            time = self.time_fn()
            if time % 10 == 0:
                matrix[1][0] += 0.05
                matrix[1][1] -= 0.05

        # Traffic dependent degradation
        if self.traffic_fn:
            traffic = self.traffic_fn()
            link_traffic = traffic.get((sender, receiver), 0.0)
            if link_traffic > 0.5:
                matrix[1][0] += 0.3
                matrix[1][1] -= 0.3

        # Clamp and Normalize matrix to stay a valid probability distribution.
        matrix = np.clip(matrix, 0.0, 1.0)
        matrix = matrix / matrix.sum(axis=1, keepdims=True)
        return matrix

    def _update_pair(self, sender: str, receiver: str, comms_matrix: ActiveCommunication):
        """
        Updates the state for a single sender-receiver communication link.
        - Retrieves the current communication state of the link (sender → receiver)
        - Adjusts the link's transition matrix based on context, time, or traffic
        - Samples the next state (connected/disconnected) from the transition matrix
        - Updates both the internal state and the communication matrix accordingly
        """
        current_state = self.state[(sender, receiver)]
        key = (sender, receiver)
        matrix = self.transition_probabilities.get(key, np.array([[0.9, 0.1],
                                                                  [0.1, 0.9]]))
        adjusted = self._adjust_matrix(matrix, sender, receiver)
        next_state = int(self.rng.choice([0, 1], p=adjusted[current_state]))
        self.state[(sender, receiver)] = next_state
        comms_matrix.update(sender, receiver, next_state == 1)

    def update_connectivity(self, comms_matrix: ActiveCommunication):
        """
        Updates the connectivity status for all agent pairs using parallel processing.

        Iterates through all directed sender-receiver pairs (excluding self-loops)
        and calls `_update_pair` for each using `joblib.Parallel` to potentially
        speed up computation on multicore machines.

        """
        if self.parallel_executor is not None:
            self.parallel_executor(
                delayed(self._update_pair)(sender, receiver, comms_matrix)
                for sender in self.agent_ids
                for receiver in self.agent_ids
                if sender != receiver
            )
        else:
            # Run sequentially without joblib to avoid pickling issues
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
