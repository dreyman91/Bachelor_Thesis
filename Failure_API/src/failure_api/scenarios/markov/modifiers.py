import numpy as np
from typing import Dict, Optional, Callable, Any


def create_weather_modifier(weather_fn: Callable[[], Dict[str, float]]) -> Callable:
    """
    Creates a Modifier function that adjusts transition probabilities based on weather conditions.

    :param weather_fn: Returns a dictionary with weather conditions.
    :return: Takes a matrix, sender, snd receiver and returns a modified matrix.
    """

    def weather_modifier(matrix: np.ndarray, sender: str, receiver: str) -> np.ndarray:
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
    def traffic_modifier(matrix: np.ndarray, sender: str, receiver: str) -> np.ndarray:
        matrix = matrix.copy()
        traffic = traffic_fn()
        link_traffic = traffic.get((sender, receiver), 0.0)

        # Apply effect
        if link_traffic > 0.5: # Higher chance of disconnection when traffic is high
            matrix[1][0] += 0.3
            matrix[1][1] -= 0.3
        return matrix
    return traffic_modifier

def create_time_modifier(time_fn: Callable[[], int])-> Callable:
    def time_modifier(matrix: np.ndarray, sender: str, receiver: str) -> np.ndarray:
        matrix = matrix.copy()
        time = time_fn()
        if time % 10 == 0:
            matrix[1][0] += 0.05
            matrix[1][1] -= 0.05
        return matrix
    return time_modifier