import warnings
from typing import List, Dict, Tuple, Optional, Callable
import numpy as np
from collections import defaultdict, deque

from .base_communication_model import CommunicationModels
from .active_communication import ActiveCommunication

class DelayBasedModel(CommunicationModels):
    """
    - Each message is delayed by a variable number of steps.
    - Messages are queued and released only when their delay has expired.
    - Messages exceeding a maximum delay threshold are dropped.
    - Model can integrate time-based delay, traffic-aware latency, and external factors.

    Features:
    - Efficient with queues (O(n^2) agent-pairs)
    - Predictable latency modeling
    - Modular enough for hybrid composition with other failure models
    """

    def __init__(self,
                 agent_ids: List[str],
                 min_delay: int,
                 max_delay: int ,
                 delay_fn: Optional[Callable] = None,
                 time_fn: Optional[Callable] = None,
                 traffic_fn: Optional[Callable] = None
                 ):
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
        self.delay_fn = delay_fn
        self.time_fn = time_fn
        self.traffic_fn = traffic_fn

        # Queue: (sender, receiver) -> deque of (delay_left, bandwidth)
        self.message_queues: Dict[Tuple[str, str], deque] = defaultdict(deque)

    def reset(self):
        self.message_queues.clear()

    def _generate_delay(self, sender: str, receiver: str)-> int:
        if self.delay_fn:
            return self.delay_fn(sender, receiver)

        delay = self.rng.integers(self.min_delay, self.max_delay + 1)

        #Time-based adjustment
        if self.time_fn:
            t = self.time_fn()
            if t % 15 == 0:
                delay += 2

        # Traffic-aware adjustment
        if self.traffic_fn:
            traffic = self.traffic_fn()
            volume = traffic.get((sender, receiver), 0.0)
            delay += int(volume * 3)
        return min(delay, self.max_delay)

    def _insert_message(self, sender_idx, receiver_idx):
        """ Inserts a new message in the queue for the specified pair"""
        sender_id = self.agent_ids[sender_idx]
        receiver_id = self.agent_ids[receiver_idx]

        delay = self._generate_delay(sender_id, receiver_id)
        if delay <= self.max_delay:
            success_flag = 1.0 if self.rng.random() > 0.1 else 0.0
            if (sender_id, receiver_id) not in self.message_queues:
                self.message_queues[(sender_id, receiver_id)] = deque()
            self.message_queues[(sender_id, receiver_id)].append((delay, success_flag))

    def update_connectivity(self, comms_matrix: ActiveCommunication):

        # 1. Decrement delay counters and apply released messages
        for (s,r), queue in list(self.message_queues.items()):
            queue = self.message_queues[(s, r)]
            new_queue = deque()
            while queue:
                delay_left, success_flag = queue.popleft()
                delay_left -= 1

                if delay_left <= 0:
                    comms_matrix.update(s, r, success_flag)
                elif delay_left <= self.max_delay:
                    new_queue.append((delay_left, success_flag)) # keep alive

            if new_queue:
                self.message_queues[(s, r)] = new_queue # else drop
            else:
                self.message_queues.pop((s, r), None)

        # 2. Insert new messages into queues
        for s_idx, sender in enumerate(self.agent_ids):
            for r_idx, receiver in enumerate(self.agent_ids):
                if sender == receiver:
                    continue # Skip self communication

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