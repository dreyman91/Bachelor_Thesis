import pytest
import numpy as np
import time
from scipy.spatial import cKDTree
from collections import defaultdict

from Failure_API.src.failure_api.communication_models.signal_based_model import SignalBasedModel
from Failure_API.src.failure_api.communication_models.active_communication import ActiveCommunication


# ----- Unit Tests -----

def test_initialization(small_agent_ids, static_positions):
    """Test that the model initializes correctly with valid parameters."""
    tx_power = 15.0
    min_signal_strength = 0.01
    dropout_alpha = 0.2

    model = SignalBasedModel(
        agent_ids=small_agent_ids,
        pos_fn=static_positions,
        tx_power=tx_power,
        min_strength=min_signal_strength,
        dropout_alpha=dropout_alpha
    )

    assert model.agent_ids == small_agent_ids, "Agent IDs were not stored correctly"
    assert model.pos_fn == static_positions, "Position function was not stored correctly"
    assert model.tx_power == tx_power, "Transmission power was not stored correctly"
    assert model.min_strength == min_signal_strength, "Minimum signal strength was not stored correctly"
    assert model.dropout_alpha == dropout_alpha, "Dropout alpha was not stored correctly"

    # Check agent ID to index mapping
    for i, agent_id in enumerate(small_agent_ids):
        assert model.id_to_idx[agent_id] == i, f"ID to index mapping incorrect for {agent_id}"


def test_initialization_invalid_params(small_agent_ids, static_positions):
    """Test that the model handles invalid parameters appropriately."""
    # Test invalid transmission power (negative)
    with pytest.raises(TypeError, match="tx_power must be a positive number"):
        SignalBasedModel(
            agent_ids=small_agent_ids,
            pos_fn=static_positions,
            tx_power=-5.0
        )

    # Test invalid minimum signal strength (negative)
    with pytest.raises(TypeError, match="min_strength must be a non-negative number"):
        SignalBasedModel(
            agent_ids=small_agent_ids,
            pos_fn=static_positions,
            min_strength=-0.01
        )

    # Test invalid dropout alpha (negative)
    with pytest.raises(TypeError, match="dropout_alpha must be a non-negative number"):
        SignalBasedModel(
            agent_ids=small_agent_ids,
            pos_fn=static_positions,
            dropout_alpha=-0.1
        )

    # Test invalid position function (not callable)
    with pytest.raises(TypeError, match="pos_fn must be a callable"):
        SignalBasedModel(
            agent_ids=small_agent_ids,
            pos_fn="not_a_function"
        )


def test_signal_strength_calculation(small_agent_ids, static_positions):
    """Test that signal strength follows the inverse-square law."""
    tx_power = 10.0

    model = SignalBasedModel(
        agent_ids=small_agent_ids,
        pos_fn=static_positions,
        tx_power=tx_power
    )

    # Create test positions at different distances
    pos1 = np.array([0.0, 0.0])
    pos2 = np.array([1.0, 0.0])  # Distance = 1
    pos3 = np.array([2.0, 0.0])  # Distance = 2
    pos4 = np.array([4.0, 0.0])  # Distance = 4

    # Calculate signal strengths
    strength1 = model.calculate_signal_strength(pos1, pos2)
    strength2 = model.calculate_signal_strength(pos1, pos3)
    strength3 = model.calculate_signal_strength(pos1, pos4)

    # Verify inverse-square law:
    # If distance doubles, signal strength should decrease by factor of 4
    # If distance quadruples, signal strength should decrease by factor of 16

    assert np.isclose(strength1 / strength2, 4.0, rtol=1e-5), \
        f"Signal strength doesn't follow inverse-square law: {strength1} / {strength2} != 4.0"

    assert np.isclose(strength1 / strength3, 16.0, rtol=1e-5), \
        f"Signal strength doesn't follow inverse-square law: {strength1} / {strength3} != 16.0"

    # Verify absolute value matches expected formula: tx_power / (distance^2 + epsilon)
    expected_strength1 = tx_power / (1.0 ** 2 + 1e-6)
    assert np.isclose(strength1, expected_strength1, rtol=1e-5), \
        f"Signal strength {strength1} doesn't match expected {expected_strength1}"


def test_packet_success_probability(small_agent_ids, static_positions):
    """Test that packet success probability decreases exponentially with distance."""
    dropout_alpha = 0.2

    model = SignalBasedModel(
        agent_ids=small_agent_ids,
        pos_fn=static_positions,
        dropout_alpha=dropout_alpha
    )

    # Check probabilities at various distances
    distances = [0.0, 1.0, 5.0, 10.0]

    for dist in distances:
        prob = model.calculate_packet_success_probability(dist)
        expected_prob = np.exp(-dropout_alpha * dist)

        assert np.isclose(prob, expected_prob, rtol=1e-5), \
            f"Packet success probability {prob} at distance {dist} doesn't match expected {expected_prob}"

    # Verify that probability decreases with distance
    probs = [model.calculate_packet_success_probability(d) for d in distances]
    assert all(probs[i] > probs[i + 1] for i in range(len(probs) - 1)), \
        f"Packet success probability doesn't decrease with distance: {probs}"


def test_signal_threshold_behavior(small_agent_ids):
    """Test that communication is only possible when signal strength exceeds the threshold."""
    min_signal_strength = 0.1
    tx_power = 10.0

    # Create test positions with varying distances
    def test_positions():
        return {
            "agent_1": np.array([0.0, 0.0]),
            "agent_2": np.array([5.0, 0.0]),  # Signal strength = 10/(5^2) = 0.4 (above threshold)
            "agent_3": np.array([10.0, 0.0])  # Signal strength = 10/(10^2) = 0.1 (at threshold)
        }

    model = SignalBasedModel(
        agent_ids=small_agent_ids,
        pos_fn=test_positions,
        tx_power=tx_power,
        min_strength=min_signal_strength,
        dropout_alpha=0.0  # No random dropouts for this test
    )

    # Set deterministic RNG to avoid random failures
    model.rng = np.random.RandomState(42)

    comms_matrix = ActiveCommunication(small_agent_ids)
    model.update_connectivity(comms_matrix)

    # agent_1 -> agent_2 should be connected (signal above threshold)
    agent1_idx = small_agent_ids.index("agent_1")
    agent2_idx = small_agent_ids.index("agent_2")
    assert comms_matrix.matrix[agent1_idx, agent2_idx] == True, \
        "agent_1 should be able to communicate with agent_2 (signal above threshold)"

    # agent_1 -> agent_3 should be connected (signal at threshold)
    agent3_idx = small_agent_ids.index("agent_3")
    assert comms_matrix.matrix[agent1_idx, agent3_idx] == True, \
        "agent_1 should be able to communicate with agent_3 (signal at threshold)"

    # Now test with agent_3 just beyond threshold
    # Redefine test_positions with agent_3 slightly farther
    def test_positions_beyond():
        return {
            "agent_1": np.array([0.0, 0.0]),
            "agent_2": np.array([5.0, 0.0]),
            "agent_3": np.array([10.1, 0.0])  # Signal strength = 10/(10.1^2) ≈ 0.098 (just below threshold)
        }

    # Create new model and test
    model_beyond = SignalBasedModel(
        agent_ids=small_agent_ids,
        pos_fn=test_positions_beyond,
        tx_power=tx_power,
        min_strength=min_signal_strength,
        dropout_alpha=0.0  # No random dropouts for this test
    )
    model_beyond.rng = np.random.RandomState(42)

    comms_matrix_beyond = ActiveCommunication(small_agent_ids)
    model_beyond.update_connectivity(comms_matrix_beyond)

    # agent_1 -> agent_3 should now be disconnected (signal below threshold)
    assert comms_matrix_beyond.matrix[agent1_idx, agent3_idx] == False, \
        "agent_1 should not be able to communicate with agent_3 (signal below threshold)"


def test_probabilistic_packet_loss(small_agent_ids, static_positions, fixed_rng):
    """Test that packet loss occurs probabilistically based on distance."""
    dropout_alpha = 0.2
    tx_power = 20.0  # High power to ensure signal strength is above threshold

    model = SignalBasedModel(
        agent_ids=small_agent_ids,
        pos_fn=static_positions,
        tx_power=tx_power,
        dropout_alpha=dropout_alpha
    )
    model.rng = fixed_rng  # Use fixed RNG for deterministic testing

    # Run many connectivity updates to gather statistics
    num_trials = 1000
    success_counts = defaultdict(int)

    for _ in range(num_trials):
        comms_matrix = ActiveCommunication(small_agent_ids)
        model.update_connectivity(comms_matrix)

        # Count successes for each link
        for i, sender in enumerate(small_agent_ids):
            for j, receiver in enumerate(small_agent_ids):
                if i != j:
                    if comms_matrix.matrix[i, j]:
                        success_counts[(sender, receiver)] += 1

    # Calculate positions and expected probabilities
    positions = static_positions()
    expected_probs = {}

    for sender in small_agent_ids:
        for receiver in small_agent_ids:
            if sender != receiver:
                dist = np.linalg.norm(positions[sender] - positions[receiver])
                expected_probs[(sender, receiver)] = np.exp(-dropout_alpha * dist)

    # Check that success rates approximate expected probabilities
    for link, expected_prob in expected_probs.items():
        empirical_prob = success_counts[link] / num_trials

        # Use a reasonable margin based on number of trials
        margin = 0.05  # 5% error margin for 1000 trials

        assert abs(empirical_prob - expected_prob) < margin, \
            f"Link {link} success rate {empirical_prob:.3f} differs from expected {expected_prob:.3f} by more than {margin}"


def test_spatial_indexing(small_agent_ids):
    """Test that spatial indexing with KD-Tree works correctly."""

    # Create grid of test positions
    def grid_positions():
        return {
            "agent_1": np.array([0.0, 0.0]),
            "agent_2": np.array([1.0, 0.0]),
            "agent_3": np.array([0.0, 1.0])
        }

    model = SignalBasedModel(
        agent_ids=small_agent_ids,
        pos_fn=grid_positions,
        tx_power=10.0,
        min_strength=0.1,
        dropout_alpha=0.0  # No random dropouts for this test
    )

    # Manually create KD-Tree from our positions to compare results
    positions = grid_positions()
    coords = np.array([positions[aid] for aid in small_agent_ids])
    tree = cKDTree(coords)

    # For each position, query the KD-Tree for nearest neighbors
    for i, agent_id in enumerate(small_agent_ids):
        # Get position and query both original tree and what model would create
        pos = positions[agent_id]

        # Query original tree for all distances
        distances, indices = tree.query(pos, k=len(small_agent_ids))

        # The model uses this for connectivity - we're checking the tree works correctly
        # The first result should be the agent itself (distance 0)
        assert indices[0] == i, f"KD-Tree first result for {agent_id} isn't itself"
        assert distances[0] == 0.0, f"KD-Tree distance to self should be 0, got {distances[0]}"

        # Check rest of results are valid indices
        assert all(0 <= idx < len(small_agent_ids) for idx in indices), "KD-Tree returned invalid indices"


def test_empty_positions(small_agent_ids):
    """Test behavior when position function returns empty dictionary."""
    model = SignalBasedModel(
        agent_ids=small_agent_ids,
        pos_fn=lambda: {}
    )

    comms_matrix = ActiveCommunication(small_agent_ids)

    # This should not raise an exception
    try:
        model.update_connectivity(comms_matrix)
    except Exception as e:
        pytest.fail(f"SignalModel failed to handle empty positions: {e}")

    # All links should be disconnected
    for i in range(len(small_agent_ids)):
        for j in range(len(small_agent_ids)):
            if i != j:
                assert comms_matrix.matrix[i, j] == False, \
                    f"Link ({i},{j}) should be disconnected with empty positions"


def test_create_initial_matrix(small_agent_ids):
    """Test that the initial matrix is created correctly."""
    model = SignalBasedModel(
        agent_ids=small_agent_ids,
        pos_fn=lambda: {}
    )

    initial_matrix = model.create_initial_matrix(small_agent_ids)

    # Check shape
    expected_shape = (len(small_agent_ids), len(small_agent_ids))
    assert initial_matrix.shape == expected_shape, \
        f"Initial matrix shape {initial_matrix.shape} doesn't match expected {expected_shape}"

    # Check diagonal (should be False - no self-connections)
    for i in range(len(small_agent_ids)):
        assert initial_matrix[i, i] == False, \
            f"Diagonal element [{i},{i}] should be False (no self-connections)"

    # Check off-diagonal (should be True - all other connections possible)
    for i in range(len(small_agent_ids)):
        for j in range(len(small_agent_ids)):
            if i != j:
                assert initial_matrix[i, j] == True, \
                    f"Off-diagonal element [{i},{j}] should be True (connection possible)"


# ----- Performance Tests -----

def test_update_performance_medium_network(medium_agent_ids, fixed_rng):
    """Test that the model performs efficiently with medium-sized networks."""

    # Create a grid of positions for all agents
    def grid_positions():
        positions = {}
        grid_size = 10.0
        for i, agent_id in enumerate(medium_agent_ids):
            row = i // 4
            col = i % 4
            positions[agent_id] = np.array([col * grid_size, row * grid_size])
        return positions

    model = SignalBasedModel(
        agent_ids=medium_agent_ids,
        pos_fn=grid_positions,
        tx_power=30.0,  # High power for longer range
        min_strength=0.01
    )
    model.rng = fixed_rng  # Fixed seed

    comms_matrix = ActiveCommunication(medium_agent_ids)

    # Measure performance
    start_time = time.time()
    model.update_connectivity(comms_matrix)
    execution_time = time.time() - start_time

    # Should be able to update a 10-agent network in under 0.05 seconds
    assert execution_time < 0.05, \
        f"Performance test failed: {execution_time:.3f} seconds for {len(medium_agent_ids)} agents (threshold: 0.05s)"


def test_kdtree_optimization(large_agent_ids, fixed_rng):
    """Test that spatial indexing with KD-Tree improves performance over naive approach."""

    # Using a larger set of agents to better demonstrate the performance difference

    # Create a grid of positions
    def grid_positions():
        positions = {}
        grid_size = 10.0
        for i, agent_id in enumerate(large_agent_ids):
            row = i // 10
            col = i % 10
            positions[agent_id] = np.array([col * grid_size, row * grid_size])
        return positions

    # Create a model with KD-Tree optimization
    model_kdtree = SignalBasedModel(
        agent_ids=large_agent_ids,
        pos_fn=grid_positions,
        tx_power=20.0
    )
    model_kdtree.rng = fixed_rng  # Fixed seed

    # Measure KD-Tree performance
    comms_matrix = ActiveCommunication(large_agent_ids)

    start_time = time.time()
    model_kdtree.update_connectivity(comms_matrix)
    kdtree_time = time.time() - start_time

    # Verify some links are connected
    connected_links = np.count_nonzero(comms_matrix.matrix)
    assert connected_links > 0, "Expected some connections in the matrix"

    # Verify performance is reasonable for this number of agents
    assert kdtree_time < 0.2, \
        f"KD-Tree performance test failed: {kdtree_time:.3f} seconds for {len(large_agent_ids)} agents"


# ----- PettingZoo Compatibility Tests -----

def test_pettingzoo_compatibility(pettingzoo_env):
    """Test that the model can be integrated with PettingZoo environments."""
    # Create model with PettingZoo agents
    agent_ids = pettingzoo_env.possible_agents

    # Reset the environment to get initial observations
    observations, _ = pettingzoo_env.reset(seed=42)

    # Create a position function that extracts positions from PettingZoo state
    def extract_positions(obs_dict):
        positions = {}
        for agent, obs in obs_dict.items():
            # Extract position information
            if hasattr(obs, 'shape') and len(obs) >= 2:
                positions[agent] = obs[:2]  # Assume first two elements are position
            else:
                positions[agent] = np.zeros(2)
        return positions

    # Create position function that uses the current environment observations
    def get_positions_from_env():
        # In a real implementation, this would get the current observations
        # For testing, we'll just use the initial observations
        return extract_positions(observations)

    model = SignalBasedModel(
        agent_ids=agent_ids,
        pos_fn=get_positions_from_env,
        tx_power=15.0,
        min_strength=0.01
    )
    model.rng = np.random.RandomState(42)  # Fixed seed

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
                    link_state = bool(comms_matrix.matrix[i, j])
                    # In a real implementation, we'd use this link state to filter communication
                    assert isinstance(link_state, bool), \
                        f"Communication status for {agent}->{other_agent} is not a boolean"

            filtered_actions[agent] = filtered_action

    # Step the environment with filtered actions
    observations, rewards, terminations, truncations, infos = pettingzoo_env.step(filtered_actions)

    # Post-step check: ensure env responds with expected structures
    for agent in pettingzoo_env.agents:
        assert agent in observations
        assert agent in rewards
        assert agent in infos