# In scenarios/signal_scenarios.py

from typing import Dict, Callable, Optional
import numpy as np
from ..communication_models.base_signal_model import BaseSignalModel


class WeatherAffectedSignalModel(BaseSignalModel):
    """
    A signal propagation model that incorporates weather effects.

    This model extends the base signal model to account for how weather
    conditions (like rain, fog, or snow) can attenuate wireless signals.
    """

    def __init__(self,
                 agent_ids,
                 position_fn,
                 weather_fn: Callable[[], Dict[str, float]],
                 tx_power: float = 15.0,
                 min_signal_strength: float = 0.01,
                 dropout_alpha: float = 0.2,
                 rain_attenuation: float = 0.5,
                 fog_attenuation: float = 0.3):
        """
        Initialize weather-affected signal model.

        Parameters:
        -----------
        agent_ids : List[str]
            List of agent identifiers
        position_fn : Callable[[], Dict[str, np.ndarray]]
            Function that returns agent positions
        weather_fn : Callable[[], Dict[str, float]]
            Function that returns weather conditions (e.g., {'rain': 0.8, 'fog': 0.2})
        tx_power : float
            Transmission power
        min_signal_strength : float
            Minimum signal strength threshold
        dropout_alpha : float
            Base packet loss parameter
        rain_attenuation : float
            How much rain affects signal strength (higher = more attenuation)
        fog_attenuation : float
            How much fog affects signal strength (higher = more attenuation)
        """
        super().__init__(agent_ids, position_fn, tx_power, min_signal_strength, dropout_alpha)
        self.weather_fn = weather_fn
        self.rain_attenuation = rain_attenuation
        self.fog_attenuation = fog_attenuation

    def calculate_signal_strength(self, sender_pos: np.ndarray, receiver_pos: np.ndarray) -> float:
        """
        Calculate weather-adjusted signal strength.

        Parameters:
        -----------
        sender_pos : np.ndarray
            Position vector of the sender
        receiver_pos : np.ndarray
            Position vector of the receiver

        Returns:
        --------
        float
            Weather-adjusted signal strength
        """
        # Get base signal strength from parent class
        base_strength = super().calculate_signal_strength(sender_pos, receiver_pos)

        try:
            # Get current weather conditions
            weather = self.weather_fn()

            # Apply weather-based signal attenuation
            rain_level = weather.get('rain', 0.0)
            fog_level = weather.get('fog', 0.0)

            # Calculate attenuation factors (exponential decay)
            rain_factor = np.exp(-self.rain_attenuation * rain_level)
            fog_factor = np.exp(-self.fog_attenuation * fog_level)

            # Apply attenuation to signal strength
            return base_strength * rain_factor * fog_factor
        except Exception as e:
            print(f"Error applying weather effects: {e}")
            return base_strength


class ObstacleAwareSignalModel(BaseSignalModel):
    """
    A signal propagation model that accounts for obstacles in the environment.

    This model extends the base signal model to simulate how obstacles
    (like walls, buildings, or terrain) can block or attenuate signals.
    """

    def __init__(self,
                 agent_ids,
                 position_fn,
                 obstacle_fn: Callable[[np.ndarray, np.ndarray], float],
                 tx_power: float = 15.0,
                 min_signal_strength: float = 0.01,
                 dropout_alpha: float = 0.2):
        """
        Initialize obstacle-aware signal model.

        Parameters:
        -----------
        agent_ids : List[str]
            List of agent identifiers
        position_fn : Callable[[], Dict[str, np.ndarray]]
            Function that returns agent positions
        obstacle_fn : Callable[[np.ndarray, np.ndarray], float]
            Function that calculates obstacle impact between two positions
            Returns attenuation factor (0.0 = complete blockage, 1.0 = no effect)
        tx_power : float
            Transmission power
        min_signal_strength : float
            Minimum signal strength threshold
        dropout_alpha : float
            Base packet loss parameter
        """
        super().__init__(agent_ids, position_fn, tx_power, min_signal_strength, dropout_alpha)
        self.obstacle_fn = obstacle_fn

    def calculate_signal_strength(self, sender_pos: np.ndarray, receiver_pos: np.ndarray) -> float:
        """
        Calculate obstacle-adjusted signal strength.

        Parameters:
        -----------
        sender_pos : np.ndarray
            Position vector of the sender
        receiver_pos : np.ndarray
            Position vector of the receiver

        Returns:
        --------
        float
            Obstacle-adjusted signal strength
        """
        # Get base signal strength from parent class
        base_strength = super().calculate_signal_strength(sender_pos, receiver_pos)

        try:
            # Get obstacle attenuation factor
            obstacle_factor = self.obstacle_fn(sender_pos, receiver_pos)

            # Apply obstacle attenuation to signal strength
            return base_strength * obstacle_factor
        except Exception as e:
            print(f"Error applying obstacle effects: {e}")
            return base_strength


class FrequencyDependentSignalModel(BaseSignalModel):
    """
    A signal propagation model that accounts for frequency-dependent effects.

    This model extends the base signal model to simulate how different
    radio frequencies propagate differently through the environment.
    """

    def __init__(self,
                 agent_ids,
                 position_fn,
                 frequency: float,  # In MHz
                 tx_power: float = 15.0,
                 min_signal_strength: float = 0.01,
                 dropout_alpha: float = 0.2):
        """
        Initialize frequency-dependent signal model.

        Parameters:
        -----------
        agent_ids : List[str]
            List of agent identifiers
        position_fn : Callable[[], Dict[str, np.ndarray]]
            Function that returns agent positions
        frequency : float
            Radio frequency in MHz
        tx_power : float
            Transmission power
        min_signal_strength : float
            Minimum signal strength threshold
        dropout_alpha : float
            Base packet loss parameter
        """
        super().__init__(agent_ids, position_fn, tx_power, min_signal_strength, dropout_alpha)
        self.frequency = frequency

    def calculate_signal_strength(self, sender_pos: np.ndarray, receiver_pos: np.ndarray) -> float:
        """
        Calculate frequency-dependent signal strength using the Friis transmission equation.

        At higher frequencies, free-space path loss increases.

        Parameters:
        -----------
        sender_pos : np.ndarray
            Position vector of the sender
        receiver_pos : np.ndarray
            Position vector of the receiver

        Returns:
        --------
        float
            Frequency-adjusted signal strength
        """
        # Calculate distance
        distance = float(np.linalg.norm(sender_pos - receiver_pos))

        # Apply Friis transmission equation, which accounts for frequency
        # Path loss increases with the square of frequency
        if distance < 1e-6:
            return self.tx_power  # Avoid division by zero

        # Constants for Friis equation
        c = 299792458  # Speed of light in m/s
        f = self.frequency * 1e6  # Convert MHz to Hz
        wavelength = c / f

        # Calculate path loss using Friis equation
        path_loss = (wavelength / (4 * np.pi * distance)) ** 2

        return self.tx_power * path_loss