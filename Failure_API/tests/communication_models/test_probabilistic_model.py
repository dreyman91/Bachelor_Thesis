import pytest
import numpy as np
import time
from collections import Counter

from Failure_API.src.failure_api.communication_models.probabilistic_model import ProbabilisticModel
from Failure_API.src.failure_api.communication_models.active_communication import ActiveCommunication


# -------------Unit Tests ------------

def test_initialization(small_agent_ids):
    """Test that the model initializes correctly with valid parameters"""

    failure_prob = 0.3
    max_bandwidth = 0.1

    model = ProbabilisticModel(
        agent_ids=small_agent_ids,
        failure_prob=failure_prob,
        max_bandwidth=max_bandwidth
    )

    assert model.agent_ids == small_agent_ids, "Agent IDs were not stored correctly"
    assert model.failure_prob == failure_prob, "Failure probability was not stored correctly"
    assert model.max_bandwidth == max_bandwidth, "Max bandwidth was not stored correctly"


def test_invalid_parameters():
    """Test that the model handles invalid parameters appropriately"""
    agent_ids = ["agent_1", "agent_2"]

    # Test invalid failure probability (negative)
    with pytest.raises(ValueError, match="Failure probability must be between 0 and 1"):
        ProbabilisticModel(agent_ids=agent_ids, failure_prob=-0.1)

    # Test invalid failure probability (greater than 1)
    with pytest.raises(ValueError, match="Failure probability must be between 0 and 1"):
        ProbabilisticModel(agent_ids=agent_ids, failure_prob=1.2)

    # Test invalid max bandwidth (negative)
    with pytest.raises(ValueError, match="Max Bandwidth must be positive"):
        ProbabilisticModel(agent_ids=agent_ids, max_bandwidth=-0.1, failure_prob=0.2)


def test_update_connectivity_deterministic(small_agent_ids, small_comms_matrix, fixed_rng):
    """Test that connectivity updates are deterministic with a fixed RNG."""

    print("Initial matrix:")
    print(small_comms_matrix.get_boolean_matrix())


    model = ProbabilisticModel(
        agent_ids=small_agent_ids,
        failure_prob=0.3,
    )
    model.rng = fixed_rng

    # Update connectivity
    model.update_connectivity(small_comms_matrix)
    actual_matrix = small_comms_matrix.get_boolean_matrix()

    expected_matrix = np.array([
        [False, True, True],
        [True, False, True],
        [False, False, False]
    ])

    print("Actual matrix:")
    print(actual_matrix.astype(int))

    assert np.array_equal(actual_matrix, expected_matrix), \
        f"Matrix with seed does not match expected pattern. \nGot:\n{actual_matrix}\nExpected:\n{expected_matrix}"


def test_update_connectivity_is_deterministic():
    ids = ['agent_1', 'agent_2', 'agent_3']
    rng = np.random.RandomState(42)

    m1 = ProbabilisticModel(agent_ids=ids, failure_prob=0.3)
    m2 = ProbabilisticModel(agent_ids=ids, failure_prob=0.3)

    m1.rng = np.random.RandomState(42)
    m2.rng = np.random.RandomState(42)

    cm1 = ActiveCommunication(ids)
    cm2 = ActiveCommunication(ids)

    m1.update_connectivity(cm1)
    m2.update_connectivity(cm2)

    assert np.array_equal(cm1.get_boolean_matrix(), cm2.get_boolean_matrix()), \
        "Connectivity updates are not deterministic with the same seed"


def test_update_connectivity_statistical(small_agent_ids):
    """
    Test that connectivity updates follow the specified probability distribution.
    This is a statistical test that verifies the model's randomness behaves correctly.
    """
    failure_prob = 0.3
    max_bandwidth = 1.0

    model = ProbabilisticModel(
        agent_ids=small_agent_ids,
        failure_prob=failure_prob,
        max_bandwidth=max_bandwidth
    )

    # Set fixed seed for reproducible statistics
    model.rng = np.random.RandomState(234)

    comms_matrix = ActiveCommunication(small_agent_ids)

    # Run many iterations to gather statistics
    start_time = time.time()
    num_iterations = 10000
    failure_counts = 0
    total_links = 0

    for _ in range(num_iterations):
        comms_matrix = ActiveCommunication(small_agent_ids)
        model.update_connectivity(comms_matrix)

        # Count failures (bandwidth == 0)
        for i, sender in enumerate(small_agent_ids):
            for j, receiver in enumerate(small_agent_ids):
                if i != j:
                    total_links += 1
                    if comms_matrix.matrix[i, j] == 0:
                        failure_counts += 1

    # Performance threshold
    execution_time = time.time() - start_time
    assert execution_time < 1.0, f"Stastical test took too long: {execution_time:.2f} seconds"

    # Calculate Empirical failure probability
    empirical_prob = failure_counts / total_links

    # Statistical validation - allow for some deviation due to randomness
    # Using a 1% error margin as a reasonable statistical tolerance
    assert abs(empirical_prob - failure_prob) < 0.01, (f"Empirical failure probability {empirical_prob:.2f} differs "
                                                       f"from expected {failure_prob:.2f} by more than 1%")


def test_update_connectivity_bandwidth(small_agent_ids):
    """Test that successful connections get the specified max bandwidth."""
    max_bandwidth = 0.75

    model = ProbabilisticModel(
        agent_ids=small_agent_ids,
        failure_prob=0.0,
        max_bandwidth=max_bandwidth
    )

    comms_matrix = ActiveCommunication(small_agent_ids)
    model.update_connectivity(comms_matrix)

    # With failure_prob=0, all connections should succeed with max_bandwidth
    for i in range(len(small_agent_ids)):
        for j in range(len(small_agent_ids)):
            if i != j:
                assert comms_matrix.matrix[i, j] == max_bandwidth, (f"Connection from {i} to {j} doesn't have expected "
                                                                    f"bandwidth {max_bandwidth}")


def test_guaranteed_failure(small_agent_ids):
    """Test behavior when failure probability is 1.0 (always fail)."""
    model = ProbabilisticModel(
        agent_ids=small_agent_ids,
        failure_prob=1.0  # Always fail
    )

    comms_matrix = ActiveCommunication(small_agent_ids)
    model.update_connectivity(comms_matrix)

    # All connections should be disabled (0.0)
    for i in range(len(small_agent_ids)):
        for j in range(len(small_agent_ids)):
            if i != j:  # Skip self-connections
                assert comms_matrix.matrix[i, j] == 0.0, \
                    f"Connection from {i} to {j} should be 0.0 with failure_prob=1.0"


def test_guaranteed_success(small_agent_ids):
    """Test behavior when failure probability is 0.0 (never fail)."""
    max_bandwidth = 0.5

    model = ProbabilisticModel(
        agent_ids=small_agent_ids,
        failure_prob=0.0,  # Never fail
        max_bandwidth=max_bandwidth
    )

    comms_matrix = ActiveCommunication(small_agent_ids)
    model.update_connectivity(comms_matrix)

    # All connections should be enabled with max_bandwidth
    for i in range(len(small_agent_ids)):
        for j in range(len(small_agent_ids)):
            if i != j:  # Skip self-connections
                assert comms_matrix.matrix[i, j] == max_bandwidth, \
                    f"Connection from {i} to {j} should be {max_bandwidth} with failure_prob=0.0"


# ----- Performance Tests -----

def test_update_performance_large_network(large_agent_ids, fixed_rng):
    """Test that the model performs efficiently with larger networks."""
    model = ProbabilisticModel(
        agent_ids=large_agent_ids,
        failure_prob=0.3
    )
    model.rng = fixed_rng

    comms_matrix = ActiveCommunication(large_agent_ids)

    # Measure performance
    start_time = time.time()
    model.update_connectivity(comms_matrix)
    execution_time = time.time() - start_time

    # Should be able to update a 50-agent network in under 0.1 seconds
    assert execution_time < 0.1, \
        f"Performance test failed: {execution_time:.3f} seconds for {len(large_agent_ids)} agents (threshold: 0.1s)"


# ----- PettingZoo Compatibility Tests -----

def test_pettingzoo_compatibility(pettingzoo_env):
    """Test that the model can be integrated with PettingZoo environments."""
    # Create model with PettingZoo agents
    agent_ids = pettingzoo_env.possible_agents
    model = ProbabilisticModel(agent_ids=agent_ids, failure_prob=0.3)
    model.rng = np.random.RandomState(42)  # Fixed seed

    # Create fresh communication matrix
    comms_matrix = ActiveCommunication(agent_ids)
    model.update_connectivity(comms_matrix)

    pettingzoo_env.reset(seed=42)

    # Run a step in the environment
    actions = {agent: pettingzoo_env.action_space(agent).sample()
               for agent in pettingzoo_env.agents}

    # Simulate communication filtering based on our model
    filtered_actions = {}
    for i, agent in enumerate(pettingzoo_env.agents):
        if agent in actions:

            # Simulate filtering communication based on connectivity
            filtered_action = actions[agent]

            # Check communication access to other agents
            for j, other_agent in enumerate(pettingzoo_env.agents):
                if agent != other_agent:
                    can_communicate = comms_matrix.get_boolean_matrix()[i, j]
                    assert isinstance(can_communicate, (bool, np.bool_)), \
                        f"Communication status for {agent}->{other_agent} is not accessible"

            filtered_actions[agent] = filtered_action

    # Step the environment with filtered actions
    observations, rewards, terminations, truncations, infos = pettingzoo_env.step(filtered_actions)

    # Post-step check: ensure env responds with expected structures
    for agent in pettingzoo_env.agents:
        assert agent in observations
        assert agent in rewards
        assert agent in infos
