import pytest
import numpy as np

from Failure_API.src.failure_api.communication_models.base_markov_model import BaseMarkovModel
from Failure_API.src.failure_api.communication_models.context_aware_markov_model import ContextAwareMarkovModel
from Failure_API.src.failure_api.communication_models.active_communication import ActiveCommunication

@pytest.fixture
def agent_ids():
    """List of agent IDs used for testing."""
    return ["agent_0", "agent_1", "agent_2"]

@pytest.fixture
def transition_probabilities():
    """Sample transition probabilities for testing."""
    return {
        ("agent_0", "agent_1"): np.array([[0.9, 0.1], [0.2, 0.8]]),
        ("agent_1", "agent_2"): np.array([[0.8, 0.2], [0.3, 0.7]])
    }

@pytest.fixture
def deterministic_rng():
    """Returns a numpy RandomState with a fixed seed for deterministic testing."""
    return np.random.RandomState(seed=42)

@pytest.fixture
def base_markov_model(agent_ids, transition_probabilities, deterministic_rng):
    """Creates a BaseMarkovModel instance with deterministic behavior."""
    model = BaseMarkovModel(agent_ids, transition_probabilities)
    model.rng = deterministic_rng
    return model

@pytest.fixture
def active_communication(agent_ids):
    """Creates an ActiveCommunication instance for testing."""
    return ActiveCommunication(agent_ids)

@pytest.fixture
def context_aware_model(agent_ids, transition_probabilities, deterministic_rng):
    """Creates a ContextAwareMarkovModel instance with deterministic behavior."""
    model = ContextAwareMarkovModel(agent_ids, transition_probabilities, [])
    model.rng = deterministic_rng
    return model

# Mock context functions for testing
@pytest.fixture
def mock_weather_function():
    """Mock weather function that returns predictable weather data."""
    def get_weather():
        return {"weather": 0.7}  # Bad weather
    return get_weather

@pytest.fixture
def mock_time_function():
    """Mock time function that returns a time divisible by 10."""
    def get_time():
        return 10
    return get_time

@pytest.fixture
def mock_traffic_function():
    """Mock traffic function that returns high traffic on a specific link."""
    def get_traffic():
        return {("agent1", "agent2"): 0.6}  # High traffic
    return get_traffic