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
            raise Warning("Negative delay values are invalid. Clipping to 0.", UserWarning)
            min_delay = max(min_delay, 0)
            max_delay = max(max_delay, 0)

        if min_delay > max_delay:
            raise Warning("min_delay > max_delay, values will be swapped.", UserWarning)
            min_delay, max_delay = max_delay, min_delay

        if message_drop_probability < 0 or message_drop_probability > 1:
            raise ValueError("message_drop_probability must be between 0 and 1.")

        self.agent_ids = agent_ids
        self.min_delay = min_delay
        self.max_delay = max_delay
        self.message_drop_probability = message_drop_probability
        self.timestep = 0

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

        return self.rng.randint(self.min_delay, self.max_delay + 1)

    def _insert_message(self, sender_idx: int, receiver_idx: int):
        """ Inserts a new message in the queue for the specified pair"""

        sender = self.agent_ids[sender_idx]
        receiver = self.agent_ids[receiver_idx]

        # Generate delay for this message
        delay = self.rng.integers(
            self.min_delay, self.max_delay + 1)

        # Success flag (1.0 for success and, 0.0 for fail)
        success_flag = 1.0 if self.rng.random() > self.message_drop_probability else 0.0

        # Add a message to the queue
        key = (sender, receiver)
        if key not in self.message_queues:
            self.message_queues[key] = deque()

        self.message_queues[key].append((delay, success_flag))

    def process_existing_messages(self, comms_matrix):
        """
        Process existing message queues, deliver expired messages.

        This method:
        1. Decrements delay counters for all queued messages
        2. Applies released messages (with delay <= 0) to update the connectivity matrix
        3. Keeps delayed messages in their queues
        """
        for (s, r), queue in list(self.message_queues.items()):
            new_queue = deque()
            messages_to_deliver = []
            while queue:
                delay_left, success_flag = queue.popleft()
                delay_left -= 1

                if delay_left <= 0:
                    # Collect messages to deliver after processing the queue
                    messages_to_deliver.append(success_flag)
                else:
                    # Message still delayed
                    new_queue.append((delay_left, success_flag))
                    comms_matrix.update(s, r, 0.0)  # Keep link inactive while delayed

            # Deliver messages after processing the queue
            for success_flag in messages_to_deliver:
                comms_matrix.update(s, r, float(success_flag))

            # Update or clear queue
            if new_queue:
                self.message_queues[(s, r)] = new_queue
            else:
                # Remove the queue completely if its empty
                # Regardless of whether it is delivered or not
                self.message_queues.pop((s, r), None)

    def queue_new_messages(self, comms_matrix):
        """
        Queue new messages based on current connectivity.

        This method:
        1. Checks which links are currently active
        2. Queues new messages for these links
        3. Sets links to inactive while messages are delayed
        """
        for s_idx, sender in enumerate(self.agent_ids):
            for r_idx, receiver in enumerate(self.agent_ids):
                if sender == receiver:
                    continue  # Skip self-communication

                # Check if communication is enabled
                if comms_matrix.matrix[s_idx, r_idx] > 0:
                    # Determine if a message should be dropped
                    if self.rng.random() < self.message_drop_probability:
                        # Message dropped - set link to inactive
                        comms_matrix.update(self.agent_ids[s_idx], self.agent_ids[r_idx], 0.0)
                    else:
                        # Queue new message
                        self._insert_message(s_idx, r_idx)
                        # Set link to inactive while message is delayed
                        comms_matrix.update(self.agent_ids[s_idx], self.agent_ids[r_idx], 0.0)

    def update_connectivity(self, comms_matrix, process_only=False, queue_only=False):
        """
        Update connectivity matrix based on message queues and timing.

        This method orchestrates the message lifecycle by calling the specialized
        processing methods in the appropriate order.

        Args:
            comms_matrix: The communication matrix to update
            process_only: Only process existing messages, don't queue new ones
            queue_only: Only queue new messages, don't process existing ones
        """
        if not process_only:
            self.queue_new_messages(comms_matrix)

        if not queue_only:
            self.process_existing_messages(comms_matrix)

        self.timestep += 1

    @staticmethod
    def create_initial_matrix(agent_ids: List[str]) -> np.ndarray:
        n = len(agent_ids)
        matrix = np.ones((n, n), dtype=bool)
        np.fill_diagonal(matrix, False)
        return matrix
