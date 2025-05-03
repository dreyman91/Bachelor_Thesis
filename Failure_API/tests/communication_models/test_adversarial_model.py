import pytest
import numpy as np
import time
from collections import defaultdict

from Failure_API.src.failure_api.communication_models.adversarial_jamming_model import BaseJammingModel
from Failure_API.src.failure_api.communication_models.active_communication import ActiveCommunication


# ----- Unit Tests -----

def test_initialization(small_agent_ids):
    """Test that the model initializes correctly with valid parameters."""
    full_block = True
    noise_strength = 0.3

    model = BaseJammingModel(
        agent_ids=small_agent_ids,
        full_block=full_block,
        noise_strength=noise_strength
    )

    assert model.agent_ids == small_agent_ids, "Agent IDs were not stored correctly"
    assert model.full_block == full_block, "Full block flag was not stored correctly"
    assert model.noise_strength == noise_strength, "Noise strength was not stored correctly"

    # Check that jamming state and log are initialized empty
    assert len(model.jamming_state) == 0, "Jamming state should be empty at initialization"
    assert len(model.jamming_log) == 0, "Jamming log should be empty at initialization"


def test_initialization_invalid_params(small_agent_ids):
    """Test that the model handles invalid parameters appropriately."""
    # Test invalid noise_strength (negative)
    with pytest.raises(ValueError, match="noise_strength must be non-negative"):
        BaseJammingModel(
            agent_ids=small_agent_ids,
            noise_strength=-0.1
        )


def test_is_jammed_base_implementation(small_agent_ids):
    """Test that the base implementation of is_jammed returns False for all links."""
    model = BaseJammingModel(
        agent_ids=small_agent_ids,
        full_block=True
    )

    # Base implementation should return False for all links
    for sender in small_agent_ids:
        for receiver in small_agent_ids:
            if sender != receiver:
                assert model.is_jammed(sender, receiver, context={}) == False, \
                    f"Base implementation should return False for all links, got True for {sender}->{receiver}"


def test_update_connectivity_no_jamming(small_agent_ids):
    """Test update_connectivity when no links are jammed."""
    model = BaseJammingModel(
        agent_ids=small_agent_ids,
        full_block=True
    )

    # Override is_jammed to always return False
    model.is_jammed = lambda sender, receiver, context=None: False

    comms_matrix = ActiveCommunication(small_agent_ids)

    # Set all links to True initially
    for i in range(len(small_agent_ids)):
        for j in range(len(small_agent_ids)):
            if i != j:
                comms_matrix.update(small_agent_ids[i], small_agent_ids[j], True)

    # Update connectivity
    model.update_connectivity(comms_matrix)

    # All links should still be True
    for i in range(len(small_agent_ids)):
        for j in range(len(small_agent_ids)):
            if i != j:
                assert comms_matrix.matrix[i, j] == True, \
                    f"Link ({i},{j}) should still be True when no jamming"


def test_update_connectivity_full_block(small_agent_ids):
    """Test update_connectivity with full blocking of jammed links."""
    model = BaseJammingModel(
        agent_ids=small_agent_ids,
        full_block=True
    )

    # Jam specific links
    jammed_links = [
        ("agent_1", "agent_2"),
        ("agent_2", "agent_3")
    ]

    # Override is_jammed to return True for specific links
    def custom_is_jammed(sender, receiver, context=None):
        return (sender, receiver) in jammed_links

    model.is_jammed = custom_is_jammed

    comms_matrix = ActiveCommunication(small_agent_ids)

    # Set all links to True initially
    for i in range(len(small_agent_ids)):
        for j in range(len(small_agent_ids)):
            if i != j:
                comms_matrix.update(small_agent_ids[i], small_agent_ids[j], True)

    # Update connectivity
    model.update_connectivity(comms_matrix)

    # Check that jammed links are blocked (False)
    for sender, receiver in jammed_links:
        sender_idx = small_agent_ids.index(sender)
        receiver_idx = small_agent_ids.index(receiver)
        assert comms_matrix.matrix[sender_idx, receiver_idx] == False, \
            f"Jammed link {sender}->{receiver} should be blocked (False)"

    # Check that non-jammed links are still True
    for i, sender in enumerate(small_agent_ids):
        for j, receiver in enumerate(small_agent_ids):
            if i != j and (sender, receiver) not in jammed_links:
                assert comms_matrix.matrix[i, j] == True, \
                    f"Non-jammed link {sender}->{receiver} should still be True"


def test_update_connectivity_degraded(small_agent_ids):
    """Test update_connectivity with signal degradation (not full blocking)."""
    noise_strength = 0.3

    model = BaseJammingModel(
        agent_ids=small_agent_ids,
        full_block=False,
        noise_strength=noise_strength
    )

    # Jam specific links
    jammed_links = [
        ("agent_1", "agent_2"),
        ("agent_2", "agent_3")
    ]

    # Override is_jammed to return True for specific links
    def custom_is_jammed(sender, receiver, context=None):
        return (sender, receiver) in jammed_links

    model.is_jammed = custom_is_jammed

    comms_matrix = ActiveCommunication(small_agent_ids)

    # Set all links to 1.0 initially
    for i in range(len(small_agent_ids)):
        for j in range(len(small_agent_ids)):
            if i != j:
                comms_matrix.update(small_agent_ids[i], small_agent_ids[j], 1.0)

    # Update connectivity
    model.update_connectivity(comms_matrix)

    # Check that jammed links are degraded to noise_strength
    for sender, receiver in jammed_links:
        sender_idx = small_agent_ids.index(sender)
        receiver_idx = small_agent_ids.index(receiver)
        assert comms_matrix.matrix[sender_idx, receiver_idx] == noise_strength, \
            f"Jammed link {sender}->{receiver} should be degraded to {noise_strength}"

    # Check that non-jammed links are still 1.0
    for i, sender in enumerate(small_agent_ids):
        for j, receiver in enumerate(small_agent_ids):
            if i != j and (sender, receiver) not in jammed_links:
                assert comms_matrix.matrix[i, j] == 1.0, \
                    f"Non-jammed link {sender}->{receiver} should still be 1.0"


def test_update_connectivity_float_return(small_agent_ids):
    """Test update_connectivity when is_jammed returns a float value."""
    model = BaseJammingModel(
        agent_ids=small_agent_ids,
        full_block=False
    )

    # Override is_jammed to return a float for specific links
    def custom_is_jammed(sender, receiver, context=None):
        if sender == "agent_1" and receiver == "agent_2":
            return 0.5  # 50% signal quality
        return False

    model.is_jammed = custom_is_jammed

    comms_matrix = ActiveCommunication(small_agent_ids)

    # Set all links to 1.0 initially
    for i in range(len(small_agent_ids)):
        for j in range(len(small_agent_ids)):
            if i != j:
                comms_matrix.update(small_agent_ids[i], small_agent_ids[j], 1.0)

    # Update connectivity
    model.update_connectivity(comms_matrix)

    # Check that the specific link returns the float value
    sender_idx = small_agent_ids.index("agent_1")
    receiver_idx = small_agent_ids.index("agent_2")
    assert comms_matrix.matrix[sender_idx, receiver_idx] == 0.5, \
        "Link agent_1->agent_2 should have value 0.5"


def test_jamming_state_tracking(small_agent_ids):
    """Test that the jamming state is properly tracked."""
    model = BaseJammingModel(
        agent_ids=small_agent_ids,
        full_block=True
    )

    # Jam specific links
    jammed_links = [
        ("agent_1", "agent_2"),
        ("agent_2", "agent_3")
    ]

    # Override is_jammed to return True for specific links
    def custom_is_jammed(sender, receiver, context=None):
        return (sender, receiver) in jammed_links

    model.is_jammed = custom_is_jammed

    comms_matrix = ActiveCommunication(small_agent_ids)

    # Update connectivity
    model.update_connectivity(comms_matrix)

    # Check that jamming state has been updated for jammed links
    for sender, receiver in jammed_links:
        assert model.jamming_state[(sender, receiver)] == True, \
            f"Jamming state for {sender}->{receiver} should be True"

    # Non-jammed links should not be in the state dictionary
    for sender in small_agent_ids:
        for receiver in small_agent_ids:
            if sender != receiver and (sender, receiver) not in jammed_links:
                assert (sender, receiver) not in model.jamming_state or model.jamming_state[
                    (sender, receiver)] == False, \
                    f"Jamming state for non-jammed link {sender}->{receiver} should not be True"


def test_jamming_log_with_time(small_agent_ids):
    """Test that jamming events are logged with time information when available."""
    model = BaseJammingModel(
        agent_ids=small_agent_ids,
        full_block=True
    )

    # Jam specific links
    jammed_links = [
        ("agent_1", "agent_2"),
        ("agent_2", "agent_3")
    ]

    # Override is_jammed to return True for specific links
    def custom_is_jammed(sender, receiver, context=None):
        return (sender, receiver) in jammed_links

    # Override _build_context to include time
    def custom_build_context():
        return {"time": 42}

    model.is_jammed = custom_is_jammed
    model._build_context = custom_build_context

    comms_matrix = ActiveCommunication(small_agent_ids)

    # Update connectivity
    model.update_connectivity(comms_matrix)

    # Check that jamming events were logged with time
    for sender, receiver in jammed_links:
        assert (sender, receiver) in model.jamming_log, \
            f"Jamming event for {sender}->{receiver} was not logged"
        assert model.jamming_log[(sender, receiver)] == [42], \
            f"Incorrect time logged for {sender}->{receiver}"


def test_get_corrupted_obs(small_agent_ids):
    """Test that observations are properly corrupted."""
    model = BaseJammingModel(
        agent_ids=small_agent_ids,
        full_block=False,
        noise_strength=0.5
    )

    # Create test observations
    obs = {
        "agent_1": np.array([1.0, 2.0, 3.0]),
        "agent_2": np.array([4.0, 5.0, 6.0]),
        "agent_3": np.array([7.0, 8.0, 9.0])
    }

    # Set fixed RNG for deterministic noise
    model.rng = np.random.RandomState(42)

    # Get corrupted observations for agent_1
    corrupted = model.get_corrupted_obs("agent_1", obs)

    # agent_1's own observation should not be corrupted
    assert np.array_equal(corrupted["agent_1"], obs["agent_1"]), \
        "Agent's own observation should not be corrupted"

    # Other agents' observations should have noise added
    assert not np.array_equal(corrupted["agent_2"], obs["agent_2"]), \
        "Other agents' observations should be corrupted"
    assert not np.array_equal(corrupted["agent_3"], obs["agent_3"]), \
        "Other agents' observations should be corrupted"

    # The noise should be proportional to noise_strength
    # With seed 42, the first few random values should be deterministic
    # We can't exactly predict the values, but we can check the rough magnitude
    agent2_diff = np.linalg.norm(corrupted["agent_2"] - obs["agent_2"])
    assert 0.1 < agent2_diff < 2.0, \
        f"Noise magnitude {agent2_diff} seems off for noise_strength {model.noise_strength}"


def test_reset_functionality(small_agent_ids):
    """Test that reset() correctly clears jamming state and log."""
    model = BaseJammingModel(
        agent_ids=small_agent_ids,
        full_block=True
    )

    # Manually set some jamming states and logs
    model.jamming_state[("agent_1", "agent_2")] = True
    model.jamming_log[("agent_1", "agent_2")] = [10, 20, 30]

    # Verify state and log have data
    assert len(model.jamming_state) > 0, "Expected jamming state to have data before reset"
    assert len(model.jamming_log) > 0, "Expected jamming log to have data before reset"

    # Reset the model
    model.reset()

    # Verify state and log are cleared
    assert len(model.jamming_state) == 0, "Jamming state should be empty after reset"
    assert len(model.jamming_log) == 0, "Jamming log should be empty after reset"


def test_create_initial_matrix(small_agent_ids):
    """Test that the initial matrix is created correctly."""
    model = BaseJammingModel(
        agent_ids=small_agent_ids,
        full_block=True
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

def test_update_performance_medium_network(medium_agent_ids):
    """Test that the model performs efficiently with medium-sized networks."""
    model = BaseJammingModel(
        agent_ids=medium_agent_ids,
        full_block=True
    )

    # Jam 20% of links randomly
    num_links = len(medium_agent_ids) * (len(medium_agent_ids) - 1)
    num_jammed = int(0.2 * num_links)

    # Create jammed link pairs
    all_links = []
    for sender in medium_agent_ids:
        for receiver in medium_agent_ids:
            if sender != receiver:
                all_links.append((sender, receiver))

    # Use fixed RNG for reproducible test
    rng = np.random.RandomState(42)
    indices = rng.permutation(len(all_links))[:num_jammed]
    jammed_links = [all_links[i] for i in indices]

    # Override is_jammed to return True for selected links
    def custom_is_jammed(sender, receiver, context=None):
        return (sender, receiver) in jammed_links

    model.is_jammed = custom_is_jammed

    comms_matrix = ActiveCommunication(medium_agent_ids)

    # Measure performance
    start_time = time.time()
    model.update_connectivity(comms_matrix)
    execution_time = time.time() - start_time

    # Should be able to update a 10-agent network in under 0.05 seconds
    assert execution_time < 0.05, \
        f"Performance test failed: {execution_time:.3f} seconds for {len(medium_agent_ids)} agents (threshold: 0.05s)"


# ----- PettingZoo Compatibility Tests -----

def test_pettingzoo_compatibility(pettingzoo_env):
    """Test that the model can be integrated with PettingZoo environments."""
    # Create model with PettingZoo agents
    agent_ids = pettingzoo_env.possible_agents
    model = BaseJammingModel(
        agent_ids=agent_ids,
        full_block=True
    )

    # Jam 20% of links randomly
    num_links = len(agent_ids) * (len(agent_ids) - 1)
    num_jammed = max(1, int(0.2 * num_links))  # At least 1 jammed link

    # Create jammed link pairs
    all_links = []
    for sender in agent_ids:
        for receiver in agent_ids:
            if sender != receiver:
                all_links.append((sender, receiver))

    # Use fixed RNG for reproducible test
    rng = np.random.RandomState(42)
    indices = rng.permutation(len(all_links))[:num_jammed]
    jammed_links = [all_links[i] for i in indices]

    # Override is_jammed to return True for selected links
    def custom_is_jammed(sender, receiver, context=None):
        return (sender, receiver) in jammed_links

    model.is_jammed = custom_is_jammed

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
            filtered_action = actions[agent].copy() if hasattr(actions[agent], 'copy') else actions[agent]

            # In a real implementation, this would modify the action based on comms_matrix
            # For testing, we just verify we can access the data we need
            for j, other_agent in enumerate(pettingzoo_env.agents):
                if agent != other_agent:
                    can_communicate = bool(comms_matrix.matrix[i, j])
                    # In a real implementation, we'd use this to filter communication components
                    assert isinstance(can_communicate, bool), \
                        f"Communication status for {agent}->{other_agent} is not a boolean"

            filtered_actions[agent] = filtered_action

    filtered_actions[agent] = filtered_action

    # Step the environment with filtered actions
    observations, rewards, terminations, truncations, infos = pettingzoo_env.step(filtered_actions)

    # Post-step check: ensure env responds with expected structures
    for agent in pettingzoo_env.agents:
        assert agent in observations
        assert agent in rewards
        assert agent in infos