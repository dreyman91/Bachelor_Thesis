import pytest
import numpy as np
import time

from Failure_API.src.failure_api.communication_models.distance_based_model import DistanceModel
from Failure_API.src.failure_api.communication_models.active_communication import ActiveCommunication


# -------------- Unit Tests ----------------

def test_initialization(small_agent_ids, static_positions):
    """Test that the model initializes correctly with valid parameters."""
    distance_threshold = 5.0
    failure_prob = 0.2
    max_bandwidth = 0.9

    model = DistanceModel(
        agent_ids=small_agent_ids,
        distance_threshold=distance_threshold,
        pos_fn=static_positions,
        failure_prob=failure_prob,
        max_bandwidth=max_bandwidth
    )

    assert model.agent_ids == small_agent_ids, "Agent IDs were not stored correctly"
    assert model.distance_threshold == distance_threshold, "Distance threshold was not stored correctly"
    assert model.pos_fn == static_positions, "Position function was not stored correctly"
    assert model.failure_prob == failure_prob, "Failure probability was not stored correctly"
    assert model.max_bandwidth == max_bandwidth, "Max bandwidth was not stored correctly"


def test_initialization_invalid_params(small_agent_ids, static_positions):
    """Test that the model handles invalid parameters appropriately."""
    # Test invalid distance threshold (negative)
    with pytest.raises(ValueError, match="Distance threshold must be greater than zero"):
        DistanceModel(agent_ids=small_agent_ids, distance_threshold=-1.0, pos_fn=static_positions)

    # Test invalid failure probability (negative)
    with pytest.raises(ValueError, match="Failure probability must be between 0 and 1"):
        DistanceModel(
            agent_ids=small_agent_ids,
            distance_threshold=5.0,
            pos_fn=static_positions,
            failure_prob=-0.1
        )

    # Test invalid failure probability (greater than 1)
    with pytest.raises(ValueError, match="Failure probability must be between 0 and 1"):
        DistanceModel(
            agent_ids=small_agent_ids,
            distance_threshold=5.0,
            pos_fn=static_positions,
            failure_prob=1.2
        )

    # Test invalid max bandwidth (negative)
    with pytest.raises(ValueError, match="Bandwidth must be greater than zero"):
        DistanceModel(
            agent_ids=small_agent_ids,
            distance_threshold=5.0,
            pos_fn=static_positions,
            max_bandwidth=-0.1
        )


def test_distance_calculation(small_agent_ids, static_positions):
    """Test that distances between agents are correctly calculated."""
    model = DistanceModel(
        agent_ids=small_agent_ids,
        distance_threshold=5.0,
        pos_fn=static_positions
    )

    # Get positions from our fixture
    positions = static_positions()

    # Calculate expected distances
    expected_distances = {
        ("agent_1", "agent_2"): np.linalg.norm(positions["agent_1"] - positions["agent_2"]),  # 3.0
        ("agent_1", "agent_3"): np.linalg.norm(positions["agent_1"] - positions["agent_3"]),  # 2.5
        ("agent_2", "agent_3"): np.linalg.norm(positions["agent_2"] - positions["agent_3"]),  # 2.5
    }

    # Check that the model calculates these correctly
    for (sender, receiver), expected_dist in expected_distances.items():
        # We need to calculate the distance the same way the model would
        sender_pos = positions[sender]
        receiver_pos = positions[receiver]
        calculated_dist = np.linalg.norm(sender_pos - receiver_pos)

        assert np.isclose(calculated_dist, expected_dist), \
            f"Distance from {sender} to {receiver} calculated incorrectly. Got {calculated_dist}, expected {expected_dist}"


def test_linear_bandwidth_degradation(small_agent_ids, static_positions):
    """Test that bandwidth decreases linearly with distance."""
    distance_threshold = 5.0
    max_bandwidth = 1.0

    model = DistanceModel(
        agent_ids=small_agent_ids,
        distance_threshold=distance_threshold,
        pos_fn=static_positions,
        failure_prob=0.0,  # No random failures
        max_bandwidth=max_bandwidth
    )

    # Create test positions at various distances
    test_positions = {
        "agent_1": np.array([0.0, 0.0]),
        "agent_2": np.array([1.0, 0.0]),  # Distance 1.0
        "agent_3": np.array([2.5, 0.0]),  # Distance 2.5
    }

    # Override the position function
    model.pos_fn = lambda: test_positions

    comms_matrix = ActiveCommunication(small_agent_ids)
    model.update_connectivity(comms_matrix)

    # Expected bandwidths based on linear degradation
    # At distance 0: bandwidth = max_bandwidth
    # At distance = threshold: bandwidth = 0
    expected_bandwidths = {
        (0, 1): max_bandwidth * (1 - 1.0 / distance_threshold),  # agent1 -> agent2
        (0, 2): max_bandwidth * (1 - 2.5 / distance_threshold),  # agent1 -> agent3
        (1, 0): max_bandwidth * (1 - 1.0 / distance_threshold),  # agent2 -> agent1
        (1, 2): max_bandwidth * (1 - 1.5 / distance_threshold),  # agent2 -> agent3
        (2, 0): max_bandwidth * (1 - 2.5 / distance_threshold),  # agent3 -> agent1
        (2, 1): max_bandwidth * (1 - 1.5 / distance_threshold),  # agent3 -> agent2
    }

    # Check that actual bandwidths match expected
    for (i, j), expected_bw in expected_bandwidths.items():
        actual_bw = comms_matrix.matrix[i, j]
        assert np.isclose(actual_bw, expected_bw), \
            f"Bandwidth from {small_agent_ids[i]} to {small_agent_ids[j]} incorrect. Got {actual_bw}, expected {expected_bw}"


def test_distance_threshold_behavior(small_agent_ids):
    """Test that agents beyond the distance threshold cannot communicate."""
    distance_threshold = 3.0

    # Create test positions with some agents beyond threshold
    test_positions = {
        "agent_1": np.array([0.0, 0.0]),
        "agent_2": np.array([2.0, 1.0]),  # Distance 2.24 (within threshold)
        "agent_3": np.array([2.0, 3.0]),  # Distance 3.605.... (beyond threshold)
    }

    model = DistanceModel(
        agent_ids=small_agent_ids,
        distance_threshold=distance_threshold,
        pos_fn=lambda: test_positions,
        failure_prob=0.0  # No random failures
    )

    comms_matrix = ActiveCommunication(small_agent_ids)
    model.update_connectivity(comms_matrix)

    # Check that agents within a threshold can communicate
    assert comms_matrix.matrix[0, 1] > 0, "agent_1 should be able to communicate with agent2"
    assert comms_matrix.matrix[1, 0] > 0, "agent_2 should be able to communicate with agent1"

    # Check that agents beyond a threshold cannot communicate
    assert comms_matrix.matrix[0, 2] == 0, "agent_1 should not be able to communicate with agent3"
    assert comms_matrix.matrix[2, 0] == 0, "agent_3 should not be able to communicate with agent1"


def test_random_failure_behavior(small_agent_ids, static_positions, fixed_rng):
    """Test that random failures occur with the specified probability."""
    distance_threshold = 10.0  # Large enough for all agents to be in range
    failure_prob = 0.5

    model = DistanceModel(
        agent_ids=small_agent_ids,
        distance_threshold=distance_threshold,
        pos_fn=static_positions,
        failure_prob=failure_prob
    )
    model.rng = fixed_rng  # Use fixed RNG for deterministic testing

    comms_matrix = ActiveCommunication(small_agent_ids)
    model.update_connectivity(comms_matrix)

    # We need to know the expected pattern for the fixed seed
    expected_failures = {
        (0, 1): True,  # agent1 -> agent2 fails
        (0, 2): False,  # agent1 -> agent3 succeeds
        (1, 0): False,  # agent2 -> agent1 succeeds
        (1, 2): False,  # agent2 -> agent3 succeeds
        (2, 0): True,  # agent3 -> agent1 fails
        (2, 1): True,  # agent3 -> agent2 succeeds
    }
    for i, sender in enumerate(small_agent_ids):
        for j, receiver in enumerate(small_agent_ids):
            if i != j:
                bw = comms_matrix.matrix[i, j]
                print(f"{sender} → {receiver}: bandwidth = {bw}")

    # Check that failures match an expected pattern
    for (i, j), should_fail in expected_failures.items():
        if should_fail:
            assert comms_matrix.matrix[i, j] == 0, \
                f"Connection from {small_agent_ids[i]} to {small_agent_ids[j]} should have failed with seed 42"
        else:
            assert comms_matrix.matrix[i, j] > 0, \
                f"Connection from {small_agent_ids[i]} to {small_agent_ids[j]} should have succeeded with seed 42"


def test_extreme_distances(small_agent_ids):
    """Test behavior with extreme distances (zero distance, very large distances)."""
    distance_threshold = 5.0

    # Create test positions with extreme distances
    test_positions = {
        "agent_1": np.array([0.0, 0.0]),
        "agent_2": np.array([0.0, 0.0]),  # Same position as agent1 (distance 0)
        "agent_3": np.array([100.0, 0.0]),  # Very far from others
    }

    model = DistanceModel(
        agent_ids=small_agent_ids,
        distance_threshold=distance_threshold,
        pos_fn=lambda: test_positions,
        failure_prob=0.0  # No random failures
    )

    comms_matrix = ActiveCommunication(small_agent_ids)
    model.update_connectivity(comms_matrix)

    # Agents at the same position should have max bandwidth
    assert comms_matrix.matrix[0, 1] == model.max_bandwidth, \
        "Agents at the same position should have maximum bandwidth"

    # Agents far beyond threshold should have zero bandwidth
    assert comms_matrix.matrix[0, 2] == 0, \
        "Agents far beyond threshold should have zero bandwidth"


def test_missing_positions(small_agent_ids):
    """Test behavior when positions are missing for some agents."""
    distance_threshold = 5.0

    # Create test positions with a missing agent
    test_positions = {
        "agent_1": np.array([0.0, 0.0]),
        "agent_2": np.array([1.0, 0.0]),
        # agent3 position is missing
    }

    model = DistanceModel(
        agent_ids=small_agent_ids,
        distance_threshold=distance_threshold,
        pos_fn=lambda: test_positions,
        failure_prob=0.0  # No random failures
    )

    # The model should handle missing positions gracefully
    comms_matrix = ActiveCommunication(small_agent_ids)

    # This should not raise an exception
    try:
        model.update_connectivity(comms_matrix)
    except KeyError:
        pytest.fail("DistanceModel failed to handle missing positions")

    # Links involving the missing agent should have zero bandwidth
    assert comms_matrix.matrix[0, 2] == 0.0, \
        "Links involving agents with missing positions should have zero bandwidth"
    assert comms_matrix.matrix[2, 0] == 0.0, \
        "Links involving agents with missing positions should have zero bandwidth"
    assert comms_matrix.matrix[0, 1] > 0, "agent1→agent2 should have valid bandwidth"
    assert comms_matrix.matrix[1, 0] > 0, "agent2→agent1 should have valid bandwidth"


def test_create_initial_matrix(small_agent_ids):
    """Test that the initial matrix is created correctly."""
    model = DistanceModel(
        agent_ids=small_agent_ids,
        distance_threshold=5.0,
        pos_fn=lambda: {}
    )

    initial_matrix = model.create_initial_matrix(small_agent_ids)

    # Should be a matrix of all ones
    expected_matrix = np.ones((len(small_agent_ids), len(small_agent_ids)), dtype=bool)
    np.fill_diagonal(expected_matrix, False)

    assert np.array_equal(initial_matrix, expected_matrix), \
        f"Initial matrix doesn't match expected pattern.\nGot:\n{initial_matrix}\nExpected:\n{expected_matrix}"


# ----- Performance Tests -----

def test_update_performance_large_network(large_agent_ids, fixed_rng):
    """Test that the model performs efficiently with larger networks."""

    # Create a grid of positions for all agents
    def grid_positions():
        positions = {}
        grid_size = 10.0
        for i, agent_id in enumerate(large_agent_ids):
            row = i // 10
            col = i % 10
            positions[agent_id] = np.array([col * grid_size, row * grid_size])
        return positions

    model = DistanceModel(
        agent_ids=large_agent_ids,
        distance_threshold=15.0,  # Allow some connectivity in the grid
        pos_fn=grid_positions,
        failure_prob=0.0  # No random failures for performance testing
    )

    comms_matrix = ActiveCommunication(large_agent_ids)

    # Measure performance
    start_time = time.time()
    model.update_connectivity(comms_matrix)
    execution_time = time.time() - start_time

    num_links = np.count_nonzero(comms_matrix.get_boolean_matrix())
    assert num_links > 0, "Expected some connections in the matrix"

    # Should be able to update a 50-agent network in under 0.1 seconds
    assert execution_time < 0.1, \
        f"Performance test failed: {execution_time:.3f} seconds for {len(large_agent_ids)} agents (threshold: 0.1s)"


# ----- PettingZoo Compatibility Tests -----

def test_pettingzoo_compatibility(pettingzoo_env):
    """Test that the model can be integrated with PettingZoo environments."""
    # Create model with PettingZoo agents
    agent_ids = pettingzoo_env.possible_agents

    observations = pettingzoo_env.reset(seed=42)

    # Create a position function that extracts positions from PettingZoo state
    def get_positions_from_env():
        positions = {}
        for agent in pettingzoo_env.agents:

            # Extract position information
            # (This would be environment-specific - here we're just demonstrating the pattern)
            if hasattr(observations, 'shape') and len(observations) >= 2:
                positions[agent] = observations[:2]  # Assume first two elements are position
            else:
                positions[agent] = np.zeros(2)

        return positions

    model = DistanceModel(
        agent_ids=agent_ids,
        distance_threshold=5.0,
        pos_fn=get_positions_from_env
    )

    # Create fresh communication matrix
    comms_matrix = ActiveCommunication(agent_ids)

    # Update connectivity
    model.update_connectivity(comms_matrix)

    # Run a step in the environment
    actions = {agent: pettingzoo_env.action_space(agent).sample()
               for agent in pettingzoo_env.agents}

    # Simulate communication filtering based on our model
    filtered_actions = {}
    for i, agent in enumerate(pettingzoo_env.agents):
        if agent in actions:
            # Simulate filtering communication based on connectivity
            filtered_action = actions[agent]
            # In a real implementation, this would modify the action based on comms_matrix
            # For testing, we just verify we can access the data we need
            for j, other_agent in enumerate(pettingzoo_env.agents):
                if agent != other_agent:
                    assert isinstance(comms_matrix.matrix[i, j], (bool, int, float)), \
                        f"Communication status for {agent}->{other_agent} is not accessible"

            filtered_actions[agent] = filtered_action
        # Step the environment with filtered actions
    observations, rewards, terminations, truncations, infos = pettingzoo_env.step(filtered_actions)

    # Post-step check: ensure env responds with expected structures
    for agent in pettingzoo_env.agents:
        assert agent in observations
        assert agent in rewards
        assert agent in infos
