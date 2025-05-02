# In scenarios/jamming_scenarios.py

from typing import List, Dict, Callable, Optional, Any
import numpy as np
from ..communication_models.base_jamming_model import BaseJammingModel


class PersistentJammingModel(BaseJammingModel):
    """
    A jamming model with state persistence, where jammed links tend to stay jammed.

    This model adds memory to the jamming process, creating a more realistic
    representation of persistent interference.
    """

    def __init__(self,
                 agent_ids: List[str],
                 jammer_fn: Callable[[str, str, Dict[str, Any]], bool],
                 p_stay: float = 0.8,
                 p_recover: float = 0.2,
                 full_block: bool = True,
                 noise_strength: float = 0.0):
        """
        Initialize persistent jamming model.

        Parameters:
        -----------
        agent_ids : List[str]
            List of agent identifiers
        jammer_fn : Callable[[str, str, Dict[str, Any]], bool]
            Function that determines if a link should be newly jammed
        p_stay : float
            Probability that a jammed link stays jammed (persistence)
        p_recover : float
            Probability that an unjammed link stays unjammed despite jamming attempt
        full_block : bool
            Whether jamming completely blocks communication
        noise_strength : float
            Degradation level when not fully blocked
        """
        super().__init__(agent_ids, full_block, noise_strength)
        self.jammer_fn = jammer_fn
        self.p_stay = p_stay
        self.p_recover = p_recover

    def is_jammed(self, sender: str, receiver: str, context: Dict[str, Any] = None) -> bool:
        """
        Determine if a link is jammed, considering persistence effects.

        Parameters:
        -----------
        sender : str
            Source agent ID
        receiver : str
            Destination agent ID
        context : Dict[str, Any], optional
            Additional context information

        Returns:
        --------
        bool
            Whether the link is jammed
        """
        context = context or {}
        key = (sender, receiver)
        previously_jammed = self.jamming_state.get(key, False)

        # State persistence logic
        if previously_jammed:
            # If already jammed, might stay jammed with probability p_stay
            if self.rng.random() < self.p_stay:
                return True
        else:
            # If not jammed, might resist jamming with probability p_recover
            if self.rng.random() < self.p_recover:
                return False

        # Otherwise, check if should be newly jammed
        try:
            return self.jammer_fn(sender, receiver, context)
        except Exception as e:
            print(f"Error in jammer function: {e}")
            return False


class ScheduleBasedJammingModel(BaseJammingModel):
    """
    A jamming model that activates based on a temporal schedule.

    This model simulates temporal patterns in jamming activity, such as
    periodic interference or time-dependent attacks.
    """

    def __init__(self,
                 agent_ids: List[str],
                 time_fn: Callable[[], int],
                 schedule_fn: Callable[[int], bool],
                 full_block: bool = True,
                 noise_strength: float = 0.0):
        """
        Initialize schedule-based jamming model.

        Parameters:
        -----------
        agent_ids : List[str]
            List of agent identifiers
        time_fn : Callable[[], int]
            Function that returns the current time
        schedule_fn : Callable[[int], bool]
            Function that determines if jamming is active at a given time
        full_block : bool
            Whether jamming completely blocks communication
        noise_strength : float
            Degradation level when not fully blocked
        """
        super().__init__(agent_ids, full_block, noise_strength)
        self.time_fn = time_fn
        self.schedule_fn = schedule_fn

    def is_jammed(self, sender: str, receiver: str, context: Dict[str, Any] = None) -> bool:
        """
        Determine if a link is jammed based on the schedule.

        Parameters:
        -----------
        sender : str
            Source agent ID
        receiver : str
            Destination agent ID
        context : Dict[str, Any], optional
            Additional context information

        Returns:
        --------
        bool
            Whether the link is jammed
        """
        try:
            current_time = self.time_fn()
            is_active = self.schedule_fn(current_time)

            if is_active and current_time not in self.jamming_log[(sender, receiver)]:
                self.jamming_log[(sender, receiver)].append(current_time)

            return is_active
        except Exception as e:
            print(f"Error in schedule-based jamming: {e}")
            return False


class ZoneBasedJammingModel(BaseJammingModel):
    """
    A jamming model that targets specific spatial zones.

    This model simulates localized interference that affects all agents
    within defined geographical areas.
    """

    def __init__(self,
                 agent_ids: List[str],
                 pos_fn: Callable[[], Dict[str, np.ndarray]],
                 zone_fn: Callable[[Dict[str, np.ndarray]], Dict[str, bool]],
                 full_block: bool = True,
                 noise_strength: float = 0.0):
        """
        Initialize zone-based jamming model.

        Parameters:
        -----------
        agent_ids : List[str]
            List of agent identifiers
        pos_fn : Callable[[], Dict[str, np.ndarray]]
            Function that returns agent positions
        zone_fn : Callable[[Dict[str, np.ndarray]], Dict[str, bool]]
            Function that determines which agents are in jammed zones
        full_block : bool
            Whether jamming completely blocks communication
        noise_strength : float
            Degradation level when not fully blocked
        """
        super().__init__(agent_ids, full_block, noise_strength)
        self.pos_fn = pos_fn
        self.zone_fn = zone_fn

    def is_jammed(self, sender: str, receiver: str, context: Dict[str, Any] = None) -> bool:
        """
        Determine if a link is jammed based on spatial zones.

        A link is jammed if either the sender or receiver is in a jammed zone.

        Parameters:
        -----------
        sender : str
            Source agent ID
        receiver : str
            Destination agent ID
        context : Dict[str, Any], optional
            Additional context information

        Returns:
        --------
        bool
            Whether the link is jammed
        """
        try:
            # Get current positions
            positions = self.pos_fn()

            # Determine which agents are in jammed zones
            jammed_zones = self.zone_fn(positions)

            # Link is jammed if either sender or receiver is in a jammed zone
            return jammed_zones.get(sender, False) or jammed_zones.get(receiver, False)
        except Exception as e:
            print(f"Error in zone-based jamming: {e}")
            return False


class TargetedJammingModel(BaseJammingModel):
    """
    A jamming model that targets specific agents.

    This model simulates focused interference that specifically
    disrupts communication involving designated agents.
    """

    def __init__(self,
                 agent_ids: List[str],
                 targeted_agents: List[str],
                 full_block: bool = True,
                 noise_strength: float = 0.0):
        """
        Initialize targeted jamming model.

        Parameters:
        -----------
        agent_ids : List[str]
            List of all agent identifiers
        targeted_agents : List[str]
            List of agents specifically targeted for jamming
        full_block : bool
            Whether jamming completely blocks communication
        noise_strength : float
            Degradation level when not fully blocked
        """
        super().__init__(agent_ids, full_block, noise_strength)
        self.targeted_agents = set(targeted_agents) if targeted_agents else set()

    def is_jammed(self, sender: str, receiver: str, context: Dict[str, Any] = None) -> bool:
        """
        Determine if a link is jammed based on targeted agents.

        A link is jammed if either the sender or receiver is a targeted agent.

        Parameters:
        -----------
        sender : str
            Source agent ID
        receiver : str
            Destination agent ID
        context : Dict[str, Any], optional
            Additional context information

        Returns:
        --------
        bool
            Whether the link is jammed
        """
        return sender in self.targeted_agents or receiver in self.targeted_agents


class CompositeJammingModel(BaseJammingModel):
    """
    A jamming model that combines multiple jamming strategies.

    This model allows researchers to compose different jamming mechanisms
    to create complex interference scenarios.
    """

    def __init__(self,
                 agent_ids: List[str],
                 jamming_models: List[BaseJammingModel],
                 combination_mode: str = "any",
                 full_block: bool = True,
                 noise_strength: float = 0.0):
        """
        Initialize composite jamming model.

        Parameters:
        -----------
        agent_ids : List[str]
            List of agent identifiers
        jamming_models : List[BaseJammingModel]
            List of jamming models to combine
        combination_mode : str
            How to combine results: "any" (jamming if any model indicates)
                                   "all" (jamming only if all models indicate)
        full_block : bool
            Whether jamming completely blocks communication
        noise_strength : float
            Degradation level when not fully blocked
        """
        super().__init__(agent_ids, full_block, noise_strength)
        self.jamming_models = jamming_models
        self.combination_mode = combination_mode

        if combination_mode not in ["any", "all"]:
            print(f"Warning: Unknown combination mode '{combination_mode}'. Using 'any'.")
            self.combination_mode = "any"

    def is_jammed(self, sender: str, receiver: str, context: Dict[str, Any] = None) -> bool:
        """
        Determine if a link is jammed based on combined models.

        Parameters:
        -----------
        sender : str
            Source agent ID
        receiver : str
            Destination agent ID
        context : Dict[str, Any], optional
            Additional context information

        Returns:
        --------
        bool
            Whether the link is jammed
        """
        context = context or {}

        # Collect results from all models
        results = []
        for model in self.jamming_models:
            try:
                results.append(model.is_jammed(sender, receiver, context))
            except Exception as e:
                print(f"Error in composite jamming model: {e}")
                results.append(False)

        # Combine results according to mode
        if not results:
            return False

        if self.combination_mode == "all":
            return all(results)
        else:  # "any" mode (default)
            return any(results)