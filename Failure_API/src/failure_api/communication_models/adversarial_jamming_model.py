from .active_communication import ActiveCommunication
from .base_communication_model import CommunicationModels
from typing import List, Callable, Dict, Tuple, Optional, Union, Any
import numpy as np
from collections import defaultdict


class BaseJammingModel(CommunicationModels):
    """A foundational model for adversarial communication jamming.

    This model simulates interference in communication networks where:
    - Communication links can be disrupted (jammed) by adversarial action
    - Jamming can either completely block communication or degrade signal quality
    - Each link can be individually targeted for jamming
    """

    def __init__(self,
                 agent_ids: List[int],
                 full_block: bool = True,
                 noise_strength: float = 0.0, ):
        """
        agent_ids : List of unique identifiers for all agents
        full_block :
            If True, jamming completely blocks communication (sets to False/0.0)
            If False, jamming degrades communication quality using noise_strength
        noise_strength : When full_block is False, the value to set for degraded connections
        """
        if  noise_strength < 0.0:
            raise ValueError("noise_strength must be non-negative.")

        super().__init__()
        self.agent_ids = agent_ids
        self.full_block = full_block
        self.noise_strength = noise_strength

        # State tracking
        self.jamming_state = defaultdict(lambda: False)
        self.jamming_log = defaultdict(list)

    def _build_context(self)-> Dict[str, Any]:
        """
        Build the current context dictionary for jamming decisions.
        :return: Dictionary containing relevant state information.
        """
        return {}

    def is_jammed(self, sender: str, receiver: str, context: Dict[str, Any]) -> bool:
        """
         Determine if a communication link is jammed.
         In the base model, no links are jammed by default. Subclasses
        should override this method to implement specific jamming strategies.

        :param sender: Source agent identifier
        :param receiver: Destination agent identifier
        :param context: Dictionary containing current positions, time, etc.
        :return: Whether a link is jammed
        """
        return False

    def get_corrupted_obs(self, agent: str, obs: dict) -> dict:
        """ Apply jamming effects to an agent's observations."""
        if not self.full_block and self.noise_strength > 0.0:
            noisy_obs = {}
            for sender, val in obs.items():
                if sender == agent:
                    noisy_obs[sender] = val
                else:
                    if isinstance(val, np.ndarray):
                        noise = self.rng.normal(0, self.noise_strength, val.shape)
                        noisy_obs[sender] = val + noise
                    else:
                        noisy_obs[sender] = val
            return noisy_obs
        return obs

    def update_connectivity(self, comms_matrix: ActiveCommunication)-> None:
        """Updates connectivity matrix with jamming effects."""
        context = self._build_context()

        # Apply jamming effects to each link
        for sender in self.agent_ids:
            for receiver in self.agent_ids:
                if sender == receiver:
                    continue

                # Check if link is jammed
                jam_result = self.is_jammed(sender, receiver, context)

                if jam_result:
                    # Update communication matrix based on jamming type
                    if isinstance(jam_result, bool):
                        value = 0.0 if self.full_block else self.noise_strength
                    else:
                        value = float(jam_result)

                    # Update communication matrix
                    comms_matrix.update(sender, receiver, value if not self.full_block else False)

                    # Update jamming state
                    self.jamming_state[(sender, receiver)] = bool(jam_result)

                    # Log jams if time information is available
                    if 'time' in context:
                        self.jamming_log[(sender, receiver)].append(context['time'])

    @staticmethod
    def create_initial_matrix(agent_ids: List[str]) -> np.ndarray:
        """
        Returns a fully connected matrix with False along the diagonal
        """
        n = len(agent_ids)
        matrix = np.ones((n, n), dtype=bool)
        np.fill_diagonal(matrix, False)
        return matrix

    def reset(self):
        """Reset the model's state"""
        self.jamming_state.clear()
        self.jamming_log.clear()