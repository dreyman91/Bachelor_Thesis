import numpy as np
from typing import Dict, Optional, Callable, Any


def create_weather_modifier(weather_fn: Callable[[], Dict[str, float]]) -> Callable:
    """
    Creates a Modifier function that adjusts transition probabilities based on weather conditions.

    :param weather_fn: returns a dictionary with weather conditions.
    :return: Takes a matrix, sender, snd receiver and returns a modified matrix.
    """

    def weather_modifier(matrix: np.ndarray, sender: str, snd: str, receiver: str) -> np.ndarray:
        """
                Modifies transition probabilities based on weather conditions.

                Args:
                    matrix: Transition probability matrix
                    sender: Source agent ID
                    receiver: Destination agent ID

                Returns:
                    Modified transition probability matrix
        """
        matrix = matrix.copy()
        ctx = weather_fn()
        if ctx.get("weather", 0.0) > 0.5:
            matrix[1][0] += 0.1
            matrix[1][1] -= 0.1
        return matrix

    return weather_modifier


def create_traffic_modifier(traffic_fn: Callable[[], Dict[tuple, float]]) -> Callable:
    """
        Creates a modifier function that adjusts transition probabilities based on traffic.
    """
    def traffic_modifier(matrix: np.ndarray, sender: str, snd: str, receiver: str) -> np.ndarray:
