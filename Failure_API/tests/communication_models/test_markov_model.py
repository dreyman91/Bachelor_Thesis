import pytest
import numpy as np
import time
from collections import defaultdict

from Failure_API.src.failure_api.communication_models.markov_model import BaseMarkovModel
from Failure_API.src.failure_api.communication_models.active_communication import ActiveCommunication


# ----- Unit Tests -----

def test_initialization(small_agent_ids, simple_transition_matrix):
    """Test that the model initializes correctly with valid parameters."""
    # Create transition probabilities dictionary
    transition_probabilities = {}
    for sender in small_agent_ids:
        for receiver in small_agent_ids:
            if sender != receiver:
                transition_probabilities[(sender, receiver)] = simple_transition_matrix.copy()

    model = BaseMarkovModel(small_agent_ids, transition_probabilities)

    # Check that agent IDs are stored correctly
    assert model.agent_ids == small_agent_ids, "Agent IDs were not stored correctly"

    # Check that transition probabilities are stored correctly
    assert model.transition_probabilities == transition_probabilities, ("Transition probabilities were not stored "
                                                                        "correctly")

    # Check that all links start in connected state (1)
    for sender in small_agent_ids:
        for receiver in small_agent_ids:
            if sender != receiver:
                assert model.state[(sender, receiver)] == 1, \
                    f"Link ({sender}->{receiver}) did not initialize in connected state"


def test_matrix_validation(small_agent_ids):
    """Test that invalid transition matrices are corrected during initialization."""
    # Matrix with rows that don't sum to 1
    invalid_matrix = np.array([[0.5, 0.2], [0.3, 0.6]])

    # Matrix with out-of-range values
    invalid_values = np.array([[-0.1, 1.1], [0.5, 0.5]])

    # Create transition probabilities with invalid matrices
    transition_probabilities = {
        ("agent_1", "agent_2"): invalid_matrix,
        ("agent_2", "agent_3"): invalid_values
    }

    # Model should correct these on initialization
    model = BaseMarkovModel(small_agent_ids, transition_probabilities)

    # Check that matrices were corrected
    matrix1 = model.get_transition_matrix("agent_1", "agent_2")
    assert np.isclose(matrix1[0].sum(), 1.0), "First row of corrected matrix doesn't sum to 1.0"
    assert np.isclose(matrix1[1].sum(), 1.0), "Second row of corrected matrix doesn't sum to 1.0"

    matrix2 = model.get_transition_matrix("agent_2", "agent_3")
    assert np.all(matrix2 >= 0.0) and np.all(matrix2 <= 1.0), \
        "Corrected matrix contains values outside valid range [0,1]"
    assert np.isclose(matrix2[0].sum(), 1.0), "First row of corrected matrix doesn't sum to 1.0"
    assert np.isclose(matrix2[1].sum(), 1.0), "Second row of corrected matrix doesn't sum to 1.0"


def test_get_transition_matrix(small_agent_ids, simple_transition_matrix):
    """Test that the model returns the correct transition matrix for each link."""
    # Create transition probabilities for specific links
    transition_probabilities = {
        ("agent_1", "agent_2"): simple_transition_matrix,
        ("agent_2", "agent_3"): np.array([[0.6, 0.4], [0.3, 0.7]])
    }

    model = BaseMarkovModel(small_agent_ids, transition_probabilities)

    # Check specified links
    assert np.array_equal(model.get_transition_matrix("agent_1", "agent_2"), simple_transition_matrix), \
        "Incorrect transition matrix for agent1->agent2"

    assert np.array_equal(model.get_transition_matrix("agent_2", "agent_3"),
                          np.array([[0.6, 0.4], [0.3, 0.7]])), \
        "Incorrect transition matrix for agent_2->agent_3"

    # Check unspecified link - should return default matrix
    default_matrix = model.get_transition_matrix("agent_3", "agent_1")
    assert default_matrix.shape == (2, 2), "Default matrix has incorrect shape"
    assert np.isclose(default_matrix.sum(axis=1)[0], 1.0), "Default matrix row 0 doesn't sum to 1.0"
    assert np.isclose(default_matrix.sum(axis=1)[1], 1.0), "Default matrix row 1 doesn't sum to 1.0"
    assert default_matrix.dtype == float
    assert np.all((default_matrix >= 0) & (default_matrix <= 1)), "Default matrix contains invalid probabilities"


def test_update_pair(small_agent_ids, simple_transition_matrix, fixed_rng):
    """Test that a single sender-receiver pair updates correctly with deterministic RNG."""
    # Create model with fixed transition matrices
    transition_probabilities = {
        ("agent_1", "agent_2"): simple_transition_matrix
    }

    model = BaseMarkovModel(small_agent_ids, transition_probabilities)
    model.rng = fixed_rng  # Use fixed RNG for deterministic testing

    # Set specific initial state
    model.state[("agent_1", "agent_2")] = 1  # Connected

    # Create communication matrix
    comms_matrix = ActiveCommunication(small_agent_ids)

    # Update the pair
    model._update_pair("agent_1", "agent_2", comms_matrix)

    # Preview what RNG will output
    expected_state = fixed_rng.choice([0, 1], p=simple_transition_matrix[1])
    print(f"\nExpected state from RNG: {expected_state}")

    assert model.state[("agent_1", "agent_2")] == expected_state, \
        f"Link state didn't update as expected with fixed seed"

    # Check communication matrix was updated to match state
    agent1_idx = small_agent_ids.index("agent_1")
    agent2_idx = small_agent_ids.index("agent_2")

    expected_comms_value = expected_state == 1  # True if connected (1), False if disconnected (0)
    assert comms_matrix.matrix[agent1_idx, agent2_idx] == expected_comms_value, \
        "Communication matrix wasn't updated to match the new state"


def test_update_connectivity(small_agent_ids, simple_transition_matrix, fixed_rng):
    """Test that connectivity updates correctly for all agent pairs."""
    # Create model with fixed transition matrices for all links
    transition_probabilities = {}
    for sender in small_agent_ids:
        for receiver in small_agent_ids:
            if sender != receiver:
                transition_probabilities[(sender, receiver)] = simple_transition_matrix.copy()

    model = BaseMarkovModel(small_agent_ids, transition_probabilities)
    model.rng = fixed_rng  # Use fixed RNG for deterministic testing

    # Create communication matrix
    comms_matrix = ActiveCommunication(small_agent_ids)

    # Update connectivity
    model.update_connectivity(comms_matrix)

    # With fixed seed, we should get deterministic results
    # We would need to know the expected pattern for the fixed seed
    # Here we're checking basic properties instead

    # Every link should have been updated (be either True or False)
    for i in range(len(small_agent_ids)):
        for j in range(len(small_agent_ids)):
            if i != j:
                assert comms_matrix.matrix[i, j] in [True, False], \
                    f"Link ({i},{j}) wasn't properly updated to a boolean value"

    # Diagonal should always be False (no self-connections)
    for i in range(len(small_agent_ids)):
        assert comms_matrix.matrix[i, i] == False, \
            f"Diagonal element ({i},{i}) should be False (no self-connections)"


def test_state_transition_statistics():
    """
    Test that state transitions follow the specified probabilities over many iterations.
    This is a statistical test that verifies the Markov property.
    """
    # Define clear transition probabilities for testing
    test_matrix = np.array([
        [0.8, 0.2],  # 80% stay disconnected, 20% become connected
        [0.3, 0.7]  # 30% become disconnected, 70% stay connected
    ])

    agent_ids = ["agent1", "agent2"]
    transitions = {("agent1", "agent2"): test_matrix}

    model = BaseMarkovModel(agent_ids, transitions)
    model.rng = np.random.RandomState(12345)  # Fixed seed for reproducible statistics

    comms_matrix = ActiveCommunication(agent_ids)

    # Track transitions for statistical validation
    transition_counts = {
        (0, 0): 0,  # Disconnected -> Disconnected
        (0, 1): 0,  # Disconnected -> Connected
        (1, 0): 0,  # Connected -> Disconnected
        (1, 1): 0  # Connected -> Connected
    }

    # Choose a specific pair to track
    test_pair = ("agent1", "agent2")

    # Run many iterations to gather statistics
    start_time = time.time()
    num_iterations = 10000
    for _ in range(num_iterations):
        prev_state = model.state[test_pair]
        model._update_pair(test_pair[0], test_pair[1], comms_matrix)
        new_state = model.state[test_pair]

        # Record the transition
        transition_counts[(prev_state, new_state)] += 1

    # Performance threshold - should complete quickly
    execution_time = time.time() - start_time
    assert execution_time < 1.0, \
        f"Statistical test took too long: {execution_time:.2f} seconds"

    # Calculate empirical transition probabilities
    # Avoid division by zero if a state was never visited
    disconnected_count = transition_counts[(0, 0)] + transition_counts[(0, 1)]
    connected_count = transition_counts[(1, 0)] + transition_counts[(1, 1)]

    if disconnected_count > 0:
        empirical_0_to_0 = transition_counts[(0, 0)] / disconnected_count
        empirical_0_to_1 = transition_counts[(0, 1)] / disconnected_count

        # Statistical validation - allow for some deviation due to randomness
        # Using a 3% error margin as a reasonable statistical tolerance
        assert abs(empirical_0_to_0 - 0.8) < 0.03, \
            f"Empirical P(0→0) = {empirical_0_to_0:.3f}, expected 0.8±0.03"
        assert abs(empirical_0_to_1 - 0.2) < 0.03, \
            f"Empirical P(0→1) = {empirical_0_to_1:.3f}, expected 0.2±0.03"

    if connected_count > 0:
        empirical_1_to_0 = transition_counts[(1, 0)] / connected_count
        empirical_1_to_1 = transition_counts[(1, 1)] / connected_count

        assert abs(empirical_1_to_0 - 0.3) < 0.03, \
            f"Empirical P(1→0) = {empirical_1_to_0:.3f}, expected 0.3±0.03"
        assert abs(empirical_1_to_1 - 0.7) < 0.03, \
            f"Empirical P(1→1) = {empirical_1_to_1:.3f}, expected 0.7±0.03"

        for k, v in transition_counts.items():
            assert v > 0, f"Transition {k} never occurred — check model randomness"


def test_different_transition_matrices():
    """Test that different links can have different transition behaviors."""
    agent_ids = ["agent_1", "agent_2", "agent_3"]

    # Create different transition matrices for different links
    reliable_link = np.array([
        [0.5, 0.5],  # 50% stay disconnected, 50% become connected (quick recovery)
        [0.1, 0.9]  # 10% become disconnected, 90% stay connected (reliable)
    ])

    unreliable_link = np.array([
        [0.9, 0.1],  # 90% stay disconnected, 10% become connected (slow recovery)
        [0.4, 0.6]  # 40% become disconnected, 60% stay connected (unreliable)
    ])

    # Create model with different link properties
    transition_probabilities = {
        ("agent_1", "agent_2"): reliable_link,
        ("agent_2", "agent_1"): reliable_link,
        ("agent_2", "agent_3"): unreliable_link,
        ("agent_3", "agent_2"): unreliable_link
    }

    model = BaseMarkovModel(agent_ids, transition_probabilities)
    model.rng = np.random.RandomState(12345)  # Fixed seed

    comms_matrix = ActiveCommunication(agent_ids)

    # Run for many iterations
    disconnected_time = defaultdict(int)
    total_time = 1000

    # Initialize all links to disconnected
    for sender in agent_ids:
        for receiver in agent_ids:
            if sender != receiver:
                model.state[(sender, receiver)] = 0  # Disconnected

    # Run simulation and track disconnection time
    for _ in range(total_time):
        model.update_connectivity(comms_matrix)

        # Count disconnected links
        for link in [("agent_1", "agent_2"), ("agent_2", "agent_3")]:
            if model.state[link] == 0:  # Disconnected
                disconnected_time[link] += 1

    # The unreliable link should be disconnected more often than the reliable link
    # We expect approximately:
    # - reliable_link: disconnected ~18% of the time (equilibrium for matrix)
    # - unreliable_link: disconnected ~80% of the time (equilibrium for matrix)
    reliable_disconnected_pct = disconnected_time[("agent_1", "agent_2")] / total_time
    unreliable_disconnected_pct = disconnected_time[("agent_2", "agent_3")] / total_time

    assert reliable_disconnected_pct < 0.30, \
        f"Reliable link disconnected {reliable_disconnected_pct:.2f} of the time (expected <0.30)"
    assert unreliable_disconnected_pct > 0.70, \
        f"Unreliable link disconnected {unreliable_disconnected_pct:.2f} of the time (expected >0.70)"
    assert unreliable_disconnected_pct > reliable_disconnected_pct + 0.3, \
        f"Unreliable link ({unreliable_disconnected_pct:.2f}) not significantly less reliable than reliable link ({reliable_disconnected_pct:.2f})"


def test_create_initial_matrix(small_agent_ids):
    """Test that the initial matrix is created correctly."""
    model = BaseMarkovModel(small_agent_ids, {})
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


def test_deterministic_behavior():
    """Test that with a fixed seed, the model behaves deterministically."""
    agent_ids = ["agent1", "agent2"]
    transitions = {("agent1", "agent2"): np.array([[0.5, 0.5], [0.5, 0.5]])}

    # Create two models with the same seed
    model1 = BaseMarkovModel(agent_ids, transitions)
    model1.rng = np.random.RandomState(12345)

    model2 = BaseMarkovModel(agent_ids, transitions)
    model2.rng = np.random.RandomState(12345)

    # Run multiple updates and verify they stay synchronized
    comms1 = ActiveCommunication(agent_ids)
    comms2 = ActiveCommunication(agent_ids)

    for _ in range(100):
        model1.update_connectivity(comms1)
        model2.update_connectivity(comms2)

        # Check that states match
        assert model1.state[("agent1", "agent2")] == model2.state[("agent1", "agent2")], \
            "Models with identical seeds produced different states"
        assert np.array_equal(comms1.matrix, comms2.matrix), \
            "Models with identical seeds produced different communication matrices"


# ----- Performance Tests -----

def test_update_performance_large_network(large_agent_ids, fixed_rng):
    """Test that the model performs efficiently with larger networks."""
    # Create default model (no specific transition matrices)
    model = BaseMarkovModel(large_agent_ids, {})
    model.rng = fixed_rng  # Fixed seed

    comms_matrix = ActiveCommunication(large_agent_ids)

    # Measure performance
    start_time = time.time()
    model.update_connectivity(comms_matrix)
    execution_time = time.time() - start_time

    # Should be able to update a 50-agent network in under 0.5 seconds
    assert execution_time < 0.5, \
        f"Performance test failed: {execution_time:.3f} seconds for {len(large_agent_ids)} agents (threshold: 0.5s)"


# ----- PettingZoo Compatibility Tests -----

def test_pettingzoo_compatibility(pettingzoo_env):
    """Test that the model can be integrated with PettingZoo environments."""
    # Create Markov model with PettingZoo agents
    agent_ids = pettingzoo_env.possible_agents
    model = BaseMarkovModel(agent_ids, {})
    model.rng = np.random.RandomState(42)  # Fixed seed

    # Create fresh communication matrix
    comms_matrix = ActiveCommunication(agent_ids)
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