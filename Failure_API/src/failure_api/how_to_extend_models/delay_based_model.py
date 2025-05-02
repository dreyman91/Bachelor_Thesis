# In scenarios/delay_scenarios.py

from typing import Callable, Dict, Optional
import numpy as np
from ..communication_models.base_delay_model import BaseDelayModel


class TimeVaryingDelayModel(BaseDelayModel):
    """
    Extension of BaseDelayModel that varies delay based on simulation time.

    This model demonstrates how to extend the base delay model to incorporate
    time-dependent behavior, such as periodic network congestion.
    """

    def __init__(self,
                 agent_ids,
                 min_delay,
                 max_delay,
                 time_fn: Callable[[], int],
                 congestion_interval: int = 15,
                 congestion_additional_delay: int = 2):
        """
        Initialize time-varying delay model.

        Parameters:
        agent_ids : List of agent identifiers
        min_delay :  Minimum delay in time steps
        max_delay : Maximum delay in time steps
        time_fn : Function that returns the current simulation time
        congestion_interval : Time interval at which congestion occurs (e.g., every 15 time steps)
        congestion_additional_delay : Additional delay added during congestion periods
        """
        super().__init__(agent_ids, min_delay, max_delay)
        self.time_fn = time_fn
        self.congestion_interval = congestion_interval
        self.congestion_additional_delay = congestion_additional_delay

    def _generate_delay(self, sender: str, receiver: str) -> int:
        """
        Generate delay considering time-dependent network congestion.
        """
        # Get base delay from parent implementation
        delay = super()._generate_delay(sender, receiver)

        # Apply time-based congestion effect
        try:
            current_time = self.time_fn()
            if current_time % self.congestion_interval == 0:
                delay += self.congestion_additional_delay
        except Exception as e:
            print(f"Error in time function: {e}")

        return min(delay, self.max_delay)


class TrafficAwareDelayModel(BaseDelayModel):
    """
    Extension of BaseDelayModel that adjusts delay based on network traffic.

    This model demonstrates how to incorporate traffic-dependent behavior,
    where delays increase as communication links become more congested.
    """

    def __init__(self,
                 agent_ids,
                 min_delay,
                 max_delay,
                 traffic_fn: Callable[[], Dict[tuple, float]],
                 traffic_multiplier: float = 3.0):
        """
        Parameters:
        agent_ids : List of agent identifiers
        min_delay : Minimum delay in time steps
        max_delay : Maximum delay in time steps
        traffic_fn : Function that returns traffic levels for each link
        traffic_multiplier : Factor to multiply traffic by to determine added delay
        """
        super().__init__(agent_ids, min_delay, max_delay)
        self.traffic_fn = traffic_fn
        self.traffic_multiplier = traffic_multiplier

    def _generate_delay(self, sender: str, receiver: str) -> int:
        """
        Generate delay considering traffic conditions.
        """
        # Get base delay from parent implementation
        delay = super()._generate_delay(sender, receiver)

        # Apply traffic-based delay
        try:
            traffic = self.traffic_fn()
            volume = traffic.get((sender, receiver), 0.0)
            delay += int(volume * self.traffic_multiplier)
        except Exception as e:
            print(f"Error in traffic function: {e}")

        return min(delay, self.max_delay)


class HybridDelayModel(BaseDelayModel):
    """
    A comprehensive delay model that combines multiple factors.

    This example shows how researchers can create complex models by
    combining multiple effects while maintaining a clean structure.
    """

    def __init__(self,
                 agent_ids,
                 min_delay,
                 max_delay,
                 time_fn: Optional[Callable[[], int]] = None,
                 traffic_fn: Optional[Callable[[], Dict[tuple, float]]] = None,
                 congestion_interval: int = 15,
                 congestion_additional_delay: int = 2,
                 traffic_multiplier: float = 3.0):
        """
        Parameters:

        agent_ids : List of agent identifiers
        min_delay : Minimum delay in time steps
        max_delay :Maximum delay in time steps
        time_fn : Function that returns the current simulation time
        traffic_fn : Function that returns traffic levels for each link
        congestion_interval : Time interval at which congestion occurs
        congestion_additional_delay : Additional delay added during congestion periods
        traffic_multiplier : Factor to multiply traffic by to determine added delay
        """
        super().__init__(agent_ids, min_delay, max_delay)
        self.time_fn = time_fn
        self.traffic_fn = traffic_fn
        self.congestion_interval = congestion_interval
        self.congestion_additional_delay = congestion_additional_delay
        self.traffic_multiplier = traffic_multiplier

    def _generate_delay(self, sender: str, receiver: str) -> int:
        """
        Generate delay considering multiple factors.
        """
        # Get base delay
        delay = super()._generate_delay(sender, receiver)

        # Apply time-based congestion effect
        if self.time_fn:
            try:
                current_time = self.time_fn()
                if current_time % self.congestion_interval == 0:
                    delay += self.congestion_additional_delay
            except Exception as e:
                print(f"Error in time function: {e}")

        # Apply traffic-based delay
        if self.traffic_fn:
            try:
                traffic = self.traffic_fn()
                volume = traffic.get((sender, receiver), 0.0)
                delay += int(volume * self.traffic_multiplier)
            except Exception as e:
                print(f"Error in traffic function: {e}")

        return min(delay, self.max_delay)