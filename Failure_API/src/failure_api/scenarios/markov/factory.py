from typing import Dict, List, Callable, Optional
import numpy as np
from failure_api.communication_models.context_aware_markov_model import ContextAwareMarkovModel
from .modifiers import create_time_modifier, create_traffic_modifier, create_weather_modifier

def create_context_aware_markov_model(
        agent_ids: List[str],
        transition_probabilities: Dict[tuple[str, str], np.ndarray],
        context_fn: Optional[Callable[[], Dict[str, float]]] = None,
        time_fn: Optional[Callable[[], Dict[tuple, float]]] = None,
        traffic_fn: Optional[Callable[[], Dict[tuple, float]]] = None,
)-> ContextAwareMarkovModel:
    """
    Creates a context-aware Markov model with the specified modifiers.

    agent_ids: List of agent identifiers
        transition_probabilities: Dictionary mapping (sender, receiver) pairs to transition matrices
        context_fn: Function that returns context information like weather
        time_fn: Function that returns the current time
        traffic_fn: Function that returns traffic information
    """
    modifiers = []

    if context_fn:
        modifiers.append(create_weather_modifier(context_fn))

    if time_fn:
        modifiers.append(create_time_modifier(time_fn))

    if traffic_fn:
        modifiers.append(create_traffic_modifier(traffic_fn))

    return ContextAwareMarkovModel(agent_ids, trasition_probabilities, modifiers)
