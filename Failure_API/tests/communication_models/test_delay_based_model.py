import pytest
import numpy as np
import time
from collections import deque

from Failure_API.src.failure_api.communication_models.delay_based_model import DelayBasedModel
from Failure_API.src.failure_api.communication_models.active_communication import ActiveCommunication


# ----- Unit Tests -----

def test_initialization(small_agent_ids):
    """Test that the model initializes correctly with valid parameters."""
    min_delay = 2
    max_delay = 5
    message_drop_probability = 0.1

    model = DelayBasedModel(
        agent_ids=small_agent_ids,
        min_delay=min_delay,
        max_delay=max_delay,
        message_drop_probability=message_drop_probability
    )

    assert model.agent_ids == small_agent_ids, "Agent IDs were not stored correctly"
    assert model.min_delay == min_delay, "Minimum delay was not stored correctly"
    assert model.max_delay == max_delay, "Maximum delay was not stored correctly"
    assert model.message_drop_probability == message_drop_probability, "Message drop probability was not stored correctly"

    # Check that message queues are initialized empty
    assert len(model.message_queues) == 0, "Message queues should be empty at initialization"


def test_initialization_invalid_params(small_agent_ids):
    """Test that the model handles invalid parameters appropriately."""
    # Test negative min_delay
    with pytest.raises(Warning, match="Negative delay values are invalid. Clipping to 0."):
        DelayBasedModel(
            agent_ids=small_agent_ids,
            min_delay=-1,
            max_delay=5
        )

    # Test negative max_delay
    with pytest.raises(Warning, match="Negative delay values are invalid. Clipping to 0."):
        DelayBasedModel(
            agent_ids=small_agent_ids,
            min_delay=1,
            max_delay=-2
        )

    # Test min_delay > max_delay
    with pytest.raises(Warning, match="min_delay > max_delay, values will be swapped."):
        DelayBasedModel(
            agent_ids=small_agent_ids,
            min_delay=5,
            max_delay=2
        )

    # Test invalid message_drop_probability (negative)
    with pytest.raises(ValueError, match="message_drop_probability must be between 0 and 1"):
        DelayBasedModel(
            agent_ids=small_agent_ids,
            min_delay=1,
            max_delay=5,
            message_drop_probability=-0.1
        )

    # Test invalid message_drop_probability (greater than 1)
    with pytest.raises(ValueError, match="message_drop_probability must be between 0 and 1"):
        DelayBasedModel(
            agent_ids=small_agent_ids,
            min_delay=1,
            max_delay=5,
            message_drop_probability=1.2
        )


def test_delay_generation(small_agent_ids, fixed_rng):
    """Test that delays are generated within the specified range."""
    min_delay = 2
    max_delay = 5

    # Confirming the rng seed
    rng = np.random.RandomState(42)
    generated_delays = rng.randint(min_delay, max_delay + 1, size=10)
    print("Generated delays:", generated_delays)

    model = DelayBasedModel(
        agent_ids=small_agent_ids,
        min_delay=min_delay,
        max_delay=max_delay
    )
    model.rng = fixed_rng  # Use fixed RNG for deterministic testing

    # Generate delays multiple times and check they're within range
    for _ in range(100):
        delay = model._generate_delay("agent_1", "agent_2")
        assert min_delay <= delay <= max_delay, \
            f"Generated delay {delay} is outside valid range [{min_delay}, {max_delay}]"

    # With fixed seed, we should get predictable results for the first few calls
    model.rng = np.random.RandomState(42)  # Reset the RNG

    expected_delays = [4, 5, 2, 4]  # Example expected values with seed 42

    for expected in expected_delays:
        delay = model._generate_delay("agent_1", "agent_2")
        assert delay == expected, \
            f"Delay with fixed seed should be {expected}, got {delay}"


def test_message_insertion(small_agent_ids, fixed_rng):
    """Test that messages are correctly inserted into the queues."""
    model = DelayBasedModel(
        agent_ids=small_agent_ids,
        min_delay=2,
        max_delay=5,
        message_drop_probability=0.0  # No drops for this test
    )
    model.rng = fixed_rng  # Use fixed RNG for deterministic testing

    # Insert a message
    sender_idx = 0  # agent_1
    receiver_idx = 1  # agent_2

    model._insert_message(sender_idx, receiver_idx)

    # Check queue contains the message
    queue_key = (small_agent_ids[sender_idx], small_agent_ids[receiver_idx])
    assert queue_key in model.message_queues, "Message was not inserted into queue"

    queue = model.message_queues[queue_key]
    assert len(queue) == 1, "Queue should contain exactly one message"

    # Check message has expected delay and success flag
    delay, success_flag = queue[0]
    assert 2 <= delay <= 5, f"Message delay {delay} outside valid range [2, 5]"
    assert success_flag == 1.0, "Message success flag should be 1.0 with drop probability 0.0"


def test_message_dropping(small_agent_ids, fixed_rng):
    """Test that messages are occasionally dropped based on the drop probability."""
    model = DelayBasedModel(
        agent_ids=small_agent_ids,
        min_delay=2,
        max_delay=5,
        message_drop_probability=0.5  # 50% drop rate
    )
    model.rng = fixed_rng  # Use fixed RNG for deterministic testing

    # Insert many messages and count success/failure
    success_count = 0
    failure_count = 0

    for _ in range(100):
        # Reset RNG to a new state each time
        model.rng = np.random.RandomState(42 + _)

        # Insert message
        model._insert_message(0, 1)

        # Get the last inserted message
        queue_key = (small_agent_ids[0], small_agent_ids[1])
        queue = model.message_queues[queue_key]
        _, success_flag = queue[-1]

        if success_flag == 1.0:
            success_count += 1
        else:
            failure_count += 1

    # With a 50% drop rate and 100 messages, we expect roughly 50 successes/failures
    # Allow some margin for randomness
    assert 40 <= success_count <= 60, \
        f"Expected ~50 successful messages, got {success_count}"
    assert 40 <= failure_count <= 60, \
        f"Expected ~50 dropped messages, got {failure_count}"


def test_message_delay_processing(small_agent_ids, fixed_rng):
    """Test that messages are properly delayed and released when delay expires."""
    model = DelayBasedModel(
        agent_ids=small_agent_ids,
        min_delay=2,
        max_delay=2,  # Fixed delay of 2 time steps for predictable testing
        message_drop_probability=0.0  # No drops
    )
    model.rng = fixed_rng  # Use fixed RNG for deterministic testing

    # Create communication matrix
    comms_matrix = ActiveCommunication(small_agent_ids)

    # Set all links as active initially
    for i in range(len(small_agent_ids)):
        for j in range(len(small_agent_ids)):
            if i != j:
                comms_matrix.update(small_agent_ids[i], small_agent_ids[j], True)

    # Step 1: Insert messages (from first update)
    model.update_connectivity(comms_matrix)

    # Check that queues have messages
    queue_count = 0
    for sender in small_agent_ids:
        for receiver in small_agent_ids:
            if sender != receiver:
                key = (sender, receiver)
                if key in model.message_queues:
                    queue_count += 1

    assert queue_count > 0, "No messages were queued after first update"

    # Initial matrix should show all links as inactive (waiting for delays)
    for i in range(len(small_agent_ids)):
        for j in range(len(small_agent_ids)):
            if i != j:
                assert comms_matrix.matrix[i, j] == False, \
                    f"Link ({i},{j}) should be inactive after first update"

    # Step 2: First time step - delay decreases by 1
    model.update_connectivity(comms_matrix)

    # Still waiting for delay to expire
    for i in range(len(small_agent_ids)):
        for j in range(len(small_agent_ids)):
            if i != j:
                assert comms_matrix.matrix[i, j] == False, \
                    f"Link ({i},{j}) should still be inactive after second update"

    # Step 3: Second time step - delay expires, messages released
    model.update_connectivity(comms_matrix, process_only=True)

    # Now some links should be active (where messages were released)
    active_link_count = 0
    for i in range(len(small_agent_ids)):
        for j in range(len(small_agent_ids)):
            if i != j and comms_matrix.matrix[i, j] > 0:
                active_link_count += 1

    assert active_link_count > 0, "No links became active after delay expired"


def test_message_queue_clearing(small_agent_ids, fixed_rng):
    """Test that message queues are cleared when messages are released or dropped."""
    model = DelayBasedModel(
        agent_ids=small_agent_ids,
        min_delay=1,
        max_delay=1,  # Fixed delay of 1 for quick testing
        message_drop_probability=0.0  # No drops
    )
    model.rng = fixed_rng  # Use fixed RNG for deterministic testing

    # Create communication matrix
    comms_matrix = ActiveCommunication(small_agent_ids)

    # Set all links as active initially
    for i in range(len(small_agent_ids)):
        for j in range(len(small_agent_ids)):
            if i != j:
                comms_matrix.update(small_agent_ids[i], small_agent_ids[j], True)

    # Step 1: Insert messages
    model.update_connectivity(comms_matrix)

    # Check that queues have messages
    assert len(model.message_queues) > 0, "No messages were queued after first update"

    # Step 2: Update again - delays expire, messages released, queues should be empty
    model.update_connectivity(comms_matrix, process_only=True)

    # Verify that all messages have been processed and queues cleared
    assert len(model.message_queues) == 0, "Message queues were not cleared after delays expired"


def test_message_order_preservation(small_agent_ids, fixed_rng):
    """Test that messages are processed in the correct order (FIFO)."""
    model = DelayBasedModel(
        agent_ids=small_agent_ids,
        min_delay=1,
        max_delay=3,
        message_drop_probability=0.0  # No drops
    )
    model.rng = fixed_rng  # Use fixed RNG for deterministic testing

    # Manually insert messages with known delays
    sender = "agent_1"
    receiver = "agent_2"
    queue_key = (sender, receiver)

    model.message_queues[queue_key] = deque([
        (3, 1.0),  # Message 1: Delay 3, Success
        (2, 1.0),  # Message 2: Delay 2, Success
        (1, 1.0)  # Message 3: Delay 1, Success
    ])

    # Create communication matrix
    comms_matrix = ActiveCommunication(small_agent_ids)

    # Update connectivity - first message with delay 1 should be released
    model.update_connectivity(comms_matrix, process_only=True)

    # Check that one message was released and others advanced
    assert len(model.message_queues[queue_key]) == 2, "Queue should have 2 messages remaining"
    assert model.message_queues[queue_key][0][0] == 2, "First message should now have delay 2"
    assert model.message_queues[queue_key][1][0] == 1, "Second message should now have delay 1"

    # Message 3 was released, should have updated the matrix
    sender_idx = small_agent_ids.index(sender)
    receiver_idx = small_agent_ids.index(receiver)
    assert comms_matrix.matrix[sender_idx, receiver_idx] == True, \
        "Matrix should be updated after message released"

    # Update again - next message released
    comms_matrix = ActiveCommunication(small_agent_ids)  # Reset matrix
    model.update_connectivity(comms_matrix, process_only=True)

    assert len(model.message_queues[queue_key]) == 1, "Queue should have 1 message remaining"
    assert model.message_queues[queue_key][0][0] == 1, "Message should now have delay 1"

    # Update again - final message released, queue should be empty
    comms_matrix = ActiveCommunication(small_agent_ids)  # Reset matrix
    model.update_connectivity(comms_matrix, process_only=True)

    assert queue_key not in model.message_queues or len(model.message_queues[queue_key]) == 0, \
        "Queue should be empty after all messages processed"


def test_reset_functionality(small_agent_ids):
    """Test that reset() correctly clears all message queues."""
    model = DelayBasedModel(
        agent_ids=small_agent_ids,
        min_delay=2,
        max_delay=5
    )

    # Manually add some message queues
    model.message_queues[("agent_1", "agent_2")] = deque([(2, 1.0)])
    model.message_queues[("agent_2", "agent_3")] = deque([(3, 0.0)])

    # Verify queues exist
    assert len(model.message_queues) == 2, "Expected 2 message queues before reset"

    # Reset the model
    model.reset()

    # Verify queues are cleared
    assert len(model.message_queues) == 0, "Message queues should be empty after reset"


def test_create_initial_matrix(small_agent_ids):
    """Test that the initial matrix is created correctly."""
    model = DelayBasedModel(
        agent_ids=small_agent_ids,
        min_delay=2,
        max_delay=5
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
    model = DelayBasedModel(
        agent_ids=medium_agent_ids,
        min_delay=1,
        max_delay=3
    )
    model.rng = fixed_rng  # Fixed seed

    comms_matrix = ActiveCommunication(medium_agent_ids)

    # Set all links as active initially
    for i in range(len(medium_agent_ids)):
        for j in range(len(medium_agent_ids)):
            if i != j:
                comms_matrix.update(medium_agent_ids[i], medium_agent_ids[j], True)

    # Measure performance
    start_time = time.time()
    model.update_connectivity(comms_matrix)
    execution_time = time.time() - start_time

    # Should be able to update a 10-agent network in under 0.05 seconds
    assert execution_time < 0.05, \
        f"Performance test failed: {execution_time:.3f} seconds for {len(medium_agent_ids)} agents (threshold: 0.05s)"


def test_long_delays_performance(small_agent_ids, fixed_rng):
    """Test performance with many messages in queues with long delays."""
    model = DelayBasedModel(
        agent_ids=small_agent_ids,
        min_delay=10,
        max_delay=50,  # Long delays
        message_drop_probability=0.0  # No drops
    )
    model.rng = fixed_rng  # Fixed seed

    comms_matrix = ActiveCommunication(small_agent_ids)

    # Set all links as active initially and update to generate queues
    for i in range(len(small_agent_ids)):
        for j in range(len(small_agent_ids)):
            if i != j:
                comms_matrix.update(small_agent_ids[i], small_agent_ids[j], True)

    # Run first update to generate messages
    model.update_connectivity(comms_matrix)

    # Verify queues were created
    assert len(model.message_queues) > 0, "Expected message queues to be created"

    # Run several more updates to accumulate messages
    for _ in range(5):
        model.update_connectivity(comms_matrix, process_only=True)

    # Measure performance with full queues
    start_time = time.time()
    model.update_connectivity(comms_matrix)
    execution_time = time.time() - start_time

    # Should still be reasonably fast
    assert execution_time < 0.02, \
        f"Performance test with long delays failed: {execution_time:.3f} seconds (threshold: 0.02s)"


# ----- PettingZoo Compatibility Tests -----

def test_pettingzoo_compatibility(pettingzoo_env):
    """Test that the model can be integrated with PettingZoo environments."""
    # Create model with PettingZoo agents
    agent_ids = pettingzoo_env.possible_agents
    model = DelayBasedModel(
        agent_ids=agent_ids,
        min_delay=1,
        max_delay=3
    )
    model.rng = np.random.RandomState(42)  # Fixed seed

    # Create fresh communication matrix
    comms_matrix = ActiveCommunication(agent_ids)

    # Set initial communication links
    for i, agent in enumerate(pettingzoo_env.agents):
        for j, other_agent in enumerate(pettingzoo_env.agents):
            if i != j:
                comms_matrix.update(agent, other_agent, True)

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

    # Step the environment with filtered actions
    observations, rewards, terminations, truncations, infos = pettingzoo_env.step(filtered_actions)

    # Post-step check: ensure env responds with expected structures
    for agent in pettingzoo_env.agents:
        assert agent in observations
        assert agent in rewards
        assert agent in infos
