import pytest
import numpy as np
from mpe2 import simple_world_comm_v3
from Failure_API.src.failure_api.communication_models.active_communication import ActiveCommunication


# -----------AGENT ID Fixtures----------#
@pytest.fixture
def small_agent_ids():
    """
    Returns a small set of agent IDs for simple tests.

    Returns: List of 3 agent IDs: ["agent_1", "agent_2", "agent_3"]
    """
    return ["agent_1", "agent_2", "agent_3"]


@pytest.fixture
def medium_agent_ids():
    """
    Returns a medium set of agent IDs for moderately complex tests.
    :return: List of 10 agent IDs: ["agent1" through "agent10"]
    """
    return [f"agent_{i}" for i in range(1, 11)]

@pytest.fixture
def large_agent_ids():
    """
    Returns a large set of agent IDs for performance tests.
    :return: List of 50 Agent IDs
    """
    return [f"agent_{i}" for i in range(50)]

#------------Random Number Generator----------#


@pytest.fixture
def fixed_rng():
    """
    Returns a numpy RandomState with a fixed seed for deterministic tests.
    :return: numpy.random.RandomState with seed 42
    """
    return np.random.RandomState(42)


# ----- Communication Matrix Fixtures -----#

@pytest.fixture
def small_comms_matrix(small_agent_ids):
    """
    Returns a fresh ActiveCommunication matrix for a small agent set.
    :returns: ActiveCommunication instance for small_agent_ids
    """
    return ActiveCommunication(small_agent_ids)


@pytest.fixture
def medium_comms_matrix(medium_agent_ids):
    """
    Returns a fresh ActiveCommunication matrix for a medium agent set.
    :returns: ActiveCommunication instance for medium_agent_ids
    """
    return ActiveCommunication(medium_agent_ids)


# ----- Position Fixtures -----#

@pytest.fixture
def static_positions(small_agent_ids):
    """
    Returns a fixed position function for deterministic testing of position-based models.
    This provides a triangular arrangement of agents in 2D space:
    - agent1 at origin (0,0)
    - agent2 at (3,0)
    - agent3 at (1.5,2)

    :returns: Function that returns a dictionary mapping agent IDs to position arrays
    """

    def get_positions():
        return {
            "agent_1": np.array([0.0, 0.0]),
            "agent_2": np.array([3.0, 0.0]),
            "agent_3": np.array([1.5, 2.0])
        }

    return get_positions


@pytest.fixture
def grid_positions(medium_agent_ids):
    """
    Returns a position function that places agents in a grid pattern for testing.
    Places 10 agents in a 3x4 grid (with 2 empty spots).
    :returns: Function that returns a dictionary mapping agent IDs to position arrays
    """

    def get_positions():
        positions = {}
        grid_size = 5.0
        for i, agent_id in enumerate(medium_agent_ids):
            row = i // 4
            col = i % 4
            positions[agent_id] = np.array([col * grid_size, row * grid_size])
        return positions

    return get_positions


# ----- Time Fixtures -----#

@pytest.fixture
def constant_time():
    """
    Returns a time function that always returns the same time (10).
    :returns: Function that returns constant time value
    """
    return lambda: 10


@pytest.fixture
def incremental_time():
    """
    Returns a time function that increases by 1 each time it's called.
    :returns: Function that returns incrementing time value
    """
    time_val = [0]  # Use list for mutable state

    def get_time():
        time_val[0] += 1
        return time_val[0]

    return get_time


# ----- Context Fixtures -----#

@pytest.fixture
def weather_context():
    """
    Returns a context function that simulates weather conditions.
    :returns: Function that returns weather conditions: {"weather": 0.7}
    """
    return lambda: {"weather": 0.7}  # Consistent "bad" weather


@pytest.fixture
def traffic_context(small_agent_ids):
    """
    Returns a traffic function that simulates network traffic.
    :returns: Function that returns traffic levels for each agent pair
    """

    def get_traffic():
        traffic = {}
        for sender in small_agent_ids:
            for receiver in small_agent_ids:
                if sender != receiver:
                    # agent1->agent2 has high traffic, others have low traffic
                    if sender == "agent_1" and receiver == "agent_2":
                        traffic[(sender, receiver)] = 0.8
                    else:
                        traffic[(sender, receiver)] = 0.2
        return traffic

    return get_traffic


# ----- PettingZoo Environment Fixtures -----

@pytest.fixture
def pettingzoo_env():
    """
    Returns a PettingZoo MPE environment with a fixed seed for deterministic testing.

    :returns: simple_world_comm_v3 environment instance
    """
    env = simple_world_comm_v3.parallel_env()
    env.reset(seed=42)
    return env


# ----- Markov Model Fixtures -----

@pytest.fixture
def simple_transition_matrix():
    """
    Returns a simple 2x2 transition matrix for Markov models.

    Matrix represents:
    - 70% chance to stay disconnected, 30% chance to become connected
    - 20% chance to become disconnected, 80% chance to stay connected

    :returns: 2x2 numpy array of transition probabilities
    """
    return np.array([
        [0.7, 0.3],  # Disconnected -> [70% stay disconnected, 30% become connected]
        [0.2, 0.8]  # Connected -> [20% become disconnected, 80% stay connected]
    ])


@pytest.fixture
def reliable_transition_matrix():
    """
    Returns a highly reliable transition matrix for Markov models.

    Matrix represents:
    - 50% chance to stay disconnected, 50% chance to become connected
    - 5% chance to become disconnected, 95% chance to stay connected

    :return: 2x2 numpy array of transition probabilities
    """
    return np.array([
        [0.5, 0.5],  # Quick recovery from disconnection
        [0.05, 0.95]  # Very likely to stay connected
    ])


@pytest.fixture
def unreliable_transition_matrix():
    """
    Returns an unreliable transition matrix for Markov models.

    Matrix represents:
    - 90% chance to stay disconnected, 10% chance to become connected
    - 40% chance to become disconnected, 60% chance to stay connected

    :return:  2x2 numpy array of transition probabilities
    """
    return np.array([
        [0.9, 0.1],  # Slow recovery from disconnection
        [0.4, 0.6]  # Often becomes disconnected
    ])


# ----- Signal Model Fixtures -----

@pytest.fixture
def signal_parameters():
    """
    Returns common parameter sets for signal-based models.
    :return: Dictionary with three parameter sets: default, strong, and weak
    """
    return {
        "default": {
            "tx_power": 15.0,
            "min_signal_strength": 0.01,
            "dropout_alpha": 0.2
        },
        "strong": {
            "tx_power": 30.0,
            "min_signal_strength": 0.005,
            "dropout_alpha": 0.1
        },
        "weak": {
            "tx_power": 5.0,
            "min_signal_strength": 0.02,
            "dropout_alpha": 0.3
        }
    }


# ----- Noise Model Fixtures -----#

@pytest.fixture
def simple_observation():
    """
    Returns a simple observation for testing noise models.

    :return: Numpy array of shape (10,) with values 0 through 9
    """
    return np.arange(10, dtype=float)


@pytest.fixture
def complex_observation():
    """
    Returns a complex dictionary observation for testing noise models.
    :return: Dictionary with mixed observation types (array, scalar, nested)
    """
    return {
        "position": np.array([1.0, 2.0, 3.0]),
        "velocity": np.array([0.1, 0.2, 0.3]),
        "health": 100,
        "ammo": 50,
        "teammates": {
            "agent_2": np.array([4.0, 5.0]),
            "agent_3": np.array([7.0, 8.0])
        }
    }


@pytest.fixture
def observation_spaces():
    """
    Returns common observation space configurations for testing noise models.
    :return: Dictionary with different space configurations
    """
    from gymnasium import spaces

    return {
        "simple_box": spaces.Box(low=-10, high=10, shape=(10,)),
        "complex_dict": spaces.Dict({
            "position": spaces.Box(low=-10, high=10, shape=(3,)),
            "velocity": spaces.Box(low=-1, high=1, shape=(3,)),
            "health": spaces.Box(low=0, high=100, shape=(1,)),
            "ammo": spaces.Discrete(100),
            "teammates": spaces.Dict({
                "agent_2": spaces.Box(low=-10, high=10, shape=(2,)),
                "agent_3": spaces.Box(low=-10, high=10, shape=(2,))
            })
        })
    }