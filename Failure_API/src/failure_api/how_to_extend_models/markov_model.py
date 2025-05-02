# In scenarios/markov_scenarios.py

from typing import Callable, Dict, Optional
import numpy as np
from ..communication_models.base_markov_model import BaseMarkovModel


class WeatherAffectedMarkovModel(BaseMarkovModel):
    """
    A Markov chain model where weather conditions affect transition probabilities.

    This extension demonstrates how environmental factors can influence
    communication reliability in a principled way.
    """

    def __init__(self,
                 agent_ids,
                 transition_probabilities,
                 weather_fn: Callable[[], Dict[str, float]],
                 weather_effect_strength: float = 0.1):
        """
        Initialize weather-affected Markov model.

        Parameters:
        -----------
        agent_ids : List[str]
            List of agent identifiers
        transition_probabilities : Dict[tuple[str, str], np.ndarray]
            Base transition matrices for each link
        weather_fn : Callable[[], Dict[str, float]]
            Function that returns weather conditions (e.g., {'severity': 0.7})
        weather_effect_strength : float
            Strength of weather effect on transition probabilities
        """
        super().__init__(agent_ids, transition_probabilities)
        self.weather_fn = weather_fn
        self.weather_effect_strength = weather_effect_strength

    def get_transition_matrix(self, sender: str, receiver: str) -> np.ndarray:
        """
        Get weather-adjusted transition matrix for a specific link.

        Parameters:
        -----------
        sender : str
            Source agent ID
        receiver : str
            Destination agent ID

        Returns:
        --------
        np.ndarray
            Weather-adjusted transition matrix
        """
        # Get base transition matrix
        matrix = super().get_transition_matrix(sender, receiver).copy()

        try:
            # Get current weather conditions
            weather = self.weather_fn()
            severity = weather.get('severity', 0.0)

            if severity > 0.5:
                # Severe weather increases disconnection probability
                # and decreases connection probability
                adjustment = self.weather_effect_strength * severity

                # Adjust transition probabilities
                matrix[1, 0] += adjustment  # Increase prob. of going from connected to disconnected
                matrix[1, 1] -= adjustment  # Decrease prob. of staying connected
                matrix[0, 1] -= adjustment / 2  # Decrease prob. of going from disconnected to connected
                matrix[0, 0] += adjustment / 2  # Increase prob. of staying disconnected

                # Ensure probabilities remain valid
                matrix = np.clip(matrix, 0.0, 1.0)
                matrix = matrix / matrix.sum(axis=1, keepdims=True)  # Normalize rows
        except Exception as e:
            print(f"Error applying weather effect: {e}")

        return matrix


class TimeVaryingMarkovModel(BaseMarkovModel):
    """
    A Markov chain model where transition probabilities vary with time.

    This extension demonstrates how temporal patterns can be incorporated
    into communication reliability models.
    """

    def __init__(self,
                 agent_ids,
                 transition_probabilities,
                 time_fn: Callable[[], int],
                 cycle_period: int = 10,
                 effect_strength: float = 0.05):
        """
        Initialize a time-varying Markov model.

        Parameters:
        -----------
        agent_ids : List[str]
            List of agent identifiers
        transition_probabilities : Dict[tuple[str, str], np.ndarray]
            Base transition matrices for each link
        time_fn : Callable[[], int]
            Function that returns the current time step
        cycle_period : int
            Number of time steps in each cycle
        effect_strength : float
            Strength of temporal effect on transition probabilities
        """
        super().__init__(agent_ids, transition_probabilities)
        self.time_fn = time_fn
        self.cycle_period = cycle_period
        self.effect_strength = effect_strength

    def get_transition_matrix(self, sender: str, receiver: str) -> np.ndarray:
        """
        Get time-adjusted transition matrix for a specific link.

        Parameters:
        -----------
        sender : str
            Source agent ID
        receiver : str
            Destination agent ID

        Returns:
        --------
        np.ndarray
            Time-adjusted transition matrix
        """
        # Get base transition matrix
        matrix = super().get_transition_matrix(sender, receiver).copy()

        try:
            # Get current time and determine cycle position
            time = self.time_fn()

            # Apply effect at specific points in the cycle
            if time % self.cycle_period == 0:
                # At these times, connections become less reliable
                matrix[1, 0] += self.effect_strength  # Increase disconnection probability
                matrix[1, 1] -= self.effect_strength  # Decrease staying-connected probability

                # Ensure probabilities remain valid
                matrix = np.clip(matrix, 0.0, 1.0)
                matrix = matrix / matrix.sum(axis=1, keepdims=True)  # Normalize rows
        except Exception as e:
            print(f"Error applying time-varying effect: {e}")

        return matrix


class TrafficAwareMarkovModel(BaseMarkovModel):
    """
    A Markov chain model where communication traffic affects reliability.

    This extension demonstrates how network congestion can influence
    connection stability in a realistic way.
    """

    def __init__(self,
                 agent_ids,
                 transition_probabilities,
                 traffic_fn: Callable[[], Dict[tuple, float]],
                 congestion_threshold: float = 0.5,
                 effect_strength: float = 0.3):
        """
        Initialize traffic-aware Markov model.

        Parameters:
        -----------
        agent_ids : List[str]
            List of agent identifiers
        transition_probabilities : Dict[tuple[str, str], np.ndarray]
            Base transition matrices for each link
        traffic_fn : Callable[[], Dict[tuple, float]]
            Function that returns traffic levels for each link
        congestion_threshold : float
            Traffic level above which congestion effects apply
        effect_strength : float
            Strength of congestion effect on transition probabilities
        """
        super().__init__(agent_ids, transition_probabilities)
        self.traffic_fn = traffic_fn
        self.congestion_threshold = congestion_threshold
        self.effect_strength = effect_strength

    def get_transition_matrix(self, sender: str, receiver: str) -> np.ndarray:
        """
        Get traffic-adjusted transition matrix for a specific link.

        Parameters:
        -----------
        sender : str
            Source agent ID
        receiver : str
            Destination agent ID

        Returns:
        --------
        np.ndarray
            Traffic-adjusted transition matrix
        """
        # Get base transition matrix
        matrix = super().get_transition_matrix(sender, receiver).copy()

        try:
            # Get current traffic conditions
            traffic = self.traffic_fn()
            link_traffic = traffic.get((sender, receiver), 0.0)

            # Apply effect when traffic exceeds threshold
            if link_traffic > self.congestion_threshold:
                # Heavy traffic increases disconnection probability
                congestion_factor = (link_traffic - self.congestion_threshold) / (1 - self.congestion_threshold)
                adjustment = self.effect_strength * congestion_factor

                matrix[1, 0] += adjustment  # Increase disconnection probability
                matrix[1, 1] -= adjustment  # Decrease staying-connected probability

                # Ensure probabilities remain valid
                matrix = np.clip(matrix, 0.0, 1.0)
                matrix = matrix / matrix.sum(axis=1, keepdims=True)  # Normalize rows
        except Exception as e:
            print(f"Error applying traffic effect: {e}")

        return matrix