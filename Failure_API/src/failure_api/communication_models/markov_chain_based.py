from typing import Callable, List, Dict, Optional
import numpy as np
from .active_communication import ActiveCommunication
from .base_communication_model import CommunicationModels
from joblib import Parallel, delayed, parallel
from collections import defaultdict

class ContextAwareMarkovModel(CommunicationModels):
    """
    A Markov-based communication model where each communication link (sender -> receiver)
    is governed by a 2-state Markov chain (Connected <-> Disconnected).

    This model supports context-aware failure dynamics, where external factors (like weather, time, etc.)
    or link traffic can influence transition probabilities.
    """
    def __init__(self,
                 agent_ids: List[str],
                 transition_probabilities: Dict[tuple[str, str], np.ndarray],
                 context_fn: Optional[Callable[[], Dict[str, float]]] = None,
                 time_fn: Optional[Callable[[], int]] = None,
                 traffic_fn: Optional[Callable[[], Dict[tuple, float]]] = None,
                 parallel_executor: Optional[Callable] = None,
                 ):
        super().__init__()
        self.agent_ids = agent_ids
        self.transition_probabilities = transition_probabilities
        self.context_fn = context_fn
        self.time_fn = time_fn
        self.traffic_fn = traffic_fn
        self.parallel_executor = parallel_executor or Parallel(n_jobs=-1)

        # State: Lazy initialization
        self.state = defaultdict(lambda: 1)

    def _adjust_matrix(self, matrix: np.ndarray, sender: str, receiver: str) -> np.ndarray:
        """
        Modify transition matrix based on context, time, or traffic conditions.

        Args:
        matrix: Original transition probability matrix
        sender: Source agent ID
        receiver: Destination agent ID

        Returns:
        Modified transition probability matrix
        """
        matrix = matrix.copy()

        # Context effects
        if self.context_fn:
            ctx = self.context_fn()
            if ctx.get("weather", 0.0) > 0.5:
                matrix[1][0] += 0.1
                matrix[1][1] -= 0.1

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
                matrix[1][0] += 0.1
                matrix[1][1] -= 0.1

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
        self.parallel_executor([
            delayed(self._update_pair)(sender, receiver, comms_matrix)
            for sender in self.agent_ids
            for receiver in self.agent_ids
            if sender != receiver
        ])

    @staticmethod
    def create_initial_matrix(agent_ids: List[str]) -> np.ndarray:
        """ Returns a matrix with maximum potential connectivity set to True
        except for self-loops which are set to False"""
        n_agents = len(agent_ids)
        matrix = np.ones((n_agents, n_agents), dtype=bool)
        np.fill_diagonal(matrix, False)
        return matrix


