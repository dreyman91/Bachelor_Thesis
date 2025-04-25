from typing import Callable, List, Dict, Optional
import numpy as np
from failure_api.communication_models.active_communication import ActiveCommunication
from failure_api.communication_models.base_markov_model import BaseMarkovModel


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
        super().__init__(agent_ids, transition_probabilities)
        self.modifier_functions = modifier_functions or []

    def get_transition_matrix(self, sender: str, receiver: str) -> np.ndarray:

        # Get the base matrix
        matrix = super().get_transition_matrix(sender, receiver).copy()

        # Apply each modifier function in a sequence
        for modifier_fn in self.modifier_functions:
            matrix = modifier_fn(matrix, sender, receiver)

        matrix = np.clip(matrix, 0.0, 1.0)
        row_sums = matrix.sum(axis=1, keepdims=True)

        # Avoid division by matrix
        row_sums = np.where(row_sums == 0.0, 1.0, row_sums)
        matrix = matrix / row_sums

        return matrix
