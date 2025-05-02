import warnings
from typing import List, Dict, Tuple, Optional, Callable
import numpy as np
from collections import defaultdict, deque

from .base_communication_model import CommunicationModels
from .active_communication import ActiveCommunication


class DelayBasedModel(CommunicationModels):
    """
    A mathematical model for message propagation with variable delays.

    This model implements a queue-based approach to message propagation where:
    - Messages are assigned a delay value when generated
    - Messages are held in queues until their delay expires
    - Messages with expired delays are delivered to their destination

    The core model uses fixed delay ranges and uniform random sampling.
    Researchers can extend this model for specific scenarios by subclassing
    and overriding the _generate_delay method.
    """

    def __init__(self,
                 agent_ids: List[str],
                 min_delay: int,
                 max_delay: int,
                 message_drop_probability: float = 0.1
                 ):
        """
         Initialize the base delay model.

         Args:
           agent_ids: List of agent identifiers
           min_delay: Minimum delay for messages
           max_delay: Maximum delay for messages (messages exceeding this are dropped)
           message_drop_probability : Probability (0.0-1.0) that a message is dropped/corrupted during transmission
        """
        super().__init__()

        if min_delay < 0 or max_delay < 0:
            warnings.warn("Negative delay values are invalid. Clipping to 0.", UserWarning)
            min_delay = max(min_delay, 0)
            max_delay = max(max_delay, 0)

        if min_delay > max_delay:
            warnings.warn("min_delay > max_delay, values will be swapped.", UserWarning)
            min_delay, max_delay = max_delay, min_delay

        self.agent_ids = agent_ids
        self.min_delay = min_delay
        self.max_delay = max_delay
        self.message_drop_probability = message_drop_probability

        # Message queues: (sender, receiver) -> deque of (delay_left, success_flag)
        self.message_queues = defaultdict(deque)

    def reset(self):
        """Reset the model state by clearing all message queues."""
        self.message_queues.clear()

    def _generate_delay(self, sender: str, receiver: str) -> int:

        """
        Generate a delay value for a message between sender and receiver.

        This core implementation samples uniformly from [min_delay, max_delay].
        Subclasses can override this method to implement scenario-specific delay generation.

        :param sender: Identifier of the sending agent
        :param receiver: Identifier of the receiving agent
        :return: Delay value in time steps
        """

        return self.rng.integers(self.min_delay, self.max_delay + 1)

    def _insert_message(self, sender_idx: int, receiver_idx: int):
        """ Inserts a new message in the queue for the specified pair"""

        sender_id = self.agent_ids[sender_idx]
        receiver_id = self.agent_ids[receiver_idx]

        # Generate delay for this message
        delay = self._generate_delay(sender_id, receiver_id)

        success_flag = 1.0 if self.rng.random() > self.message_drop_probability else 0.0

        # Add message to the queue
        if (sender_id, receiver_id) not in self.message_queues:
            self.message_queues[(sender_id, receiver_id)] = deque()
        self.message_queues[(sender_id, receiver_id)].append((delay, success_flag))

    def update_connectivity(self, comms_matrix: ActiveCommunication):
        """
        Update connectivity matrix based on message queues and timing.

        This method:
        1. Decrements delay counters for all queued messages
        2. Applies released messages (with delay <= 0) to update the connectivity matrix
        3. Inserts new messages into queues based on current connectivity state
        """
        # 1. Process existing messages: Decrement delay counters and apply released messages
        for (s, r), queue in list(self.message_queues.items()):
            queue = self.message_queues[(s, r)]
            new_queue = deque()
            while queue:
                delay_left, success_flag = queue.popleft()
                delay_left -= 1

                if delay_left <= 0:
                    comms_matrix.update(s, r, success_flag)
                elif delay_left <= self.max_delay:
                    new_queue.append((delay_left, success_flag))  # keep alive

            # Update or clear queue
            if new_queue:
                self.message_queues[(s, r)] = new_queue  # else drop
            else:
                self.message_queues.pop((s, r), None)

        # 2. Insert new messages into queues based on current connectivity
        for s_idx, sender in enumerate(self.agent_ids):
            for r_idx, receiver in enumerate(self.agent_ids):
                if sender == receiver:
                    continue  # Skip self-communication

                # Check if communication is enabled
                if comms_matrix.matrix[s_idx, r_idx] > 0:
                    if self.rng.random() < comms_matrix.matrix[s_idx, r_idx]:
                        self._insert_message(s_idx, r_idx)

    @staticmethod
    def create_initial_matrix(agent_ids: List[str]) -> np.ndarray:
        n = len(agent_ids)
        matrix = np.ones((n, n), dtype=bool)
        np.fill_diagonal(matrix, False)
        return matrix
