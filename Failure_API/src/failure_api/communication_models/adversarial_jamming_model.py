from .active_communication import ActiveCommunication
from .base_communication_model import CommunicationModels
from typing import List, Callable, Dict, Tuple, Optional, Union, Any
import numpy as np
from collections import defaultdict

class AdversarialJammingModel(CommunicationModels):
    """
    A communication model simulating adversarial jamming and interference.
    Features:
    - Temporal jamming: Schedule-based activation patterns
    - Spatial jamming: Zone-based effects targeting specific regions
    - Targeted jamming: Affects specific agents regardless of position
    - Configurable effects: Complete blocking or signal degradation
    - Composable: Can be layered with other communication models
    - Extensible: Supports custom jamming strategies via flexible callable

    """
    def __init__(self,
                 agent_ids: List[str],
                 jammer_fn: Optional[Callable[[str, str, Optional[np.ndarray],
                                               Optional[np.ndarray],
                                               Optional[int]], Union[bool, float]]] = None,
                 jam_schedule_fn: Optional[Callable[[], bool]] = None,
                 jamming_log: Optional[Dict[Tuple[str, str], List[int]]] = None,
                 jam_zone_fn: Optional[Callable[[], Dict[str, bool]]] = None,
                 targeted_agents: Optional[List[str]] = None,
                 pos_fn: Optional[Callable[[], Dict[str, np.ndarray]]] = None,
                 time_fn: Optional[Callable[[], int]] = None,
                 full_block: bool = True,
                 noise_strength: float = 0.0,
                 p_stay: float = 0.8,
                 p_recover: float = 0.2,
                 p_start: float = 0.1,
                 ):
        """
        Args:
            agent_ids: List of all agent identifiers
            base_model: Optional underlying communication model to extend with jamming
            jammer_fn: Advanced option - a callable that determines if a link is jammed
                       Signature: jammer_fn(sender_id, receiver_id, sender_pos, receiver_pos, current_time) -> bool/float
            jam_schedule_fn: Simple option - function that returns True if jamming is active at current step
            jam_zone_fn: Simple option - function that returns which agents are currently in a jammed zone
            targeted_agents: Simple option - list of agents that are always targeted
            pos_fn: Function that returns current agent positions (required for position-based jamming)
            time_fn: Function that returns current simulation time (required for time-based jamming)
            full_block: If True, sets communication to 0/False. If False, applies noise_strength
            noise_strength: If not full_block, applies this degraded bandwidth value
            """
        super().__init__()
        self.agent_ids = agent_ids
        self.full_block = full_block
        self.noise_strength = noise_strength
        self.targeted_agents = set(targeted_agents) if targeted_agents else set()

        # Optional functional hooks
        self.jammer_fn = jammer_fn
        self.jam_schedule_fn = jam_schedule_fn
        self.jam_zone_fn = jam_zone_fn
        self.pos_fn = pos_fn
        self.time_fn = time_fn

        # Probability dynamics
        self.p_stay_jammed = p_stay
        self.p_recover = p_recover
        self.p_start_jam = p_start

        # State tracking
        self.jamming_state = defaultdict(lambda: 0)
        self.jamming_log = defaultdict(list) if jamming_log is None else jamming_log

        if jammer_fn and any([jam_schedule_fn, jam_zone_fn, targeted_agents]):
            print("Warning: Both jammer_fn and simple jamming options provided. jammer_fn will take precedence")

        if jammer_fn is None and all(x is None for x in [jam_schedule_fn, jam_zone_fn]) and not targeted_agents:
            print("Warning: No jamming mechanism specified. Model will have no effect.")

        if (hasattr(jammer_fn, "__code__")
                and "pos" in jammer_fn.__code__.co_varnames and self.pos_fn is None):
            print("Warning: jammer_fn seems to need position, but pos_fn was not provided.")

        if (hasattr(jammer_fn, "__code__")
                and "time" in jammer_fn.__code__.co_varnames and self.time_fn is None):
            print("Warning: jammer_fn seems to need time, but time_fn was not provided.")

    def _is_jammed(self, sender: str, receiver: str, context: Dict[str, Any]) -> bool:
        """
        Determine if a communication link is jammed based on configuration.

        Args:
            sender: Source agent ID
            receiver: Destination agent ID
            context: Dictionary containing current positions, time, etc.

        Returns:
            Either a boolean (jammed/not jammed) or a float (signal strength) depending on configuration
        """
        if self.jammer_fn:
            try:
                return self.jammer_fn(
                    sender,
                    receiver,
                    context.get('positions', {}).get(sender),
                    context.get('positions', {}).get(receiver),
                    context.get('time'),
                )

            except Exception as e:
                print(f"Error in jammer_fn for {sender}->{receiver}: {e}")
                return False



        # Handling state persistence
        key = (sender, receiver)
        previously_jammed = self.jamming_state.get(key, False)

        if previously_jammed:
            # 60% chance of staying jammed
            if self.rng.random() < self.p_stay_jammed:
                self.jamming_state[key] = True
                return True
            else:
                self.jamming_state[key] = False
                return False

        is_jammed = False

        # Schedule based  jamming
        if not previously_jammed and  self.jam_schedule_fn:
            try:
                if self.jam_schedule_fn():
                    is_jammed = True
            except Exception as e:
                print(f"Error in jam_schedule_fn: {e}")

        # Spatial jamming
        if self.jam_zone_fn and not is_jammed:
            try:
                zones = self.jam_zone_fn()
                if zones.get(sender, False) or zones.get(receiver, False):
                    is_jammed = True
            except Exception as e:
                print(f"Error in jam_zone_fn: {e}")

        # Targeted Jamming
        if not is_jammed:
            if self.targeted_agents:
                if sender in self.targeted_agents or receiver in self.targeted_agents:
                    is_jammed = True

        self.jamming_state[key] = is_jammed
        return is_jammed

    def get_corrupted_obs(self, agent:str, obs:dict)-> dict:
        """ Corrupt the incoming observation dict for a given agent if jamming applies."""
        if not self.full_block and self.noise_strength > 0.0:
            noisy_obs = {}
            for sender, val in obs.items():
                if sender == agent:
                    noisy_obs[sender] = val
                else:
                    if isinstance(val, np.ndarray):
                        noise = self.rng.normal(0, self.noise_strength, size=val.shape)
                        noisy_obs[sender] = val + noise
                    else:
                        noisy_obs[sender] = val
            return noisy_obs
        return obs

    def _update_jamming_states(self):
        """
        Updates the internal memory of which links are currently jammed.
        """
        context = {}
        if self.pos_fn:
            context['positions'] = self.pos_fn()
        if self.time_fn:
            context['time'] = self.time_fn()

        for sender in self.agent_ids:
            for receiver in self.agent_ids:
                if sender == receiver:
                    continue

                # Status
                is_jammed_now = self._is_jammed(sender, receiver, context)
                self.jamming_state[(sender, receiver)] = is_jammed_now

    def update_connectivity(self, comms_matrix: ActiveCommunication)->None:
        """ Update connectivity matrix with jamming effects."""
        #Update memory
        self._update_jamming_states()

        context = {}
        if self.pos_fn:
            context['positions'] = self.pos_fn()
        if self.time_fn:
            context['time'] = self.time_fn()

        # Apply jamming effects
        for sender in self.agent_ids:
            for receiver in self.agent_ids:
                if sender == receiver:
                    continue

                jam_result = self._is_jammed(sender, receiver, context)

                if jam_result:
                    if isinstance(jam_result, bool):
                        value = 0.0 if self.full_block else self.noise_strength
                    else:
                        value = jam_result

                    comms_matrix.update(sender, receiver, value if not self.full_block else False)
                    self.jamming_state[(sender, receiver)] = bool(jam_result)

                    # Log jams
                    if self.time_fn:
                        current_time = self.time_fn()
                        self.jamming_log[(sender, receiver)].append(current_time)

    @staticmethod
    def create_initial_matrix(agent_ids: List[str]) -> np.ndarray:
        """
        Returns a fully connected matrix ith False along the diagonal
        """
        n = len(agent_ids)
        matrix = np.ones((n, n), dtype=bool)
        np.fill_diagonal(matrix, False)
        return matrix

    def reset(self):
        self.jamming_log.clear()




