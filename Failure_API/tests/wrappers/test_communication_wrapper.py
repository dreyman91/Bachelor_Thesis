import pytest
import numpy as np
from mpe2 import simple_world_comm_v3
from gymnasium import spaces

from Failure_API.src.failure_api.wrappers.communication_wrapper import CommunicationWrapper
from Failure_API.src.failure_api.communication_models import (ProbabilisticModel, DistanceModel, ActiveCommunication)


# ----- Test Fixtures -----

@pytest.fixture
def test_env():
    """Create a simple test environment."""
    env = simple_world_comm_v3.env(num_good=3, num_adversaries=0, num_obstacles=0, max_cycles=25)
    env.reset(seed=42)
    return env


@pytest.fixture
def probabilistic_model(test_env):
    """Create a probabilistic failure model."""
    return ProbabilisticModel(
        agent_ids=test_env.possible_agents,
        failure_prob=0.99
    )


@pytest.fixture
def distance_model(test_env):
    """Create a distance-based failure model."""

    # Create a simple position function for testing
    def position_fn():
        return {
            agent: np.array([i, i]) for i, agent in enumerate(test_env.possible_agents)
        }

    return DistanceModel(
        agent_ids=test_env.possible_agents,
        distance_threshold=1.5,
        pos_fn=position_fn
    )


# ----- Unit Tests -----

def test_initialization(test_env, probabilistic_model):
    """Test that the wrapper initializes correctly."""
    wrapper = CommunicationWrapper(
        env=test_env,
        failure_models=[probabilistic_model]
    )

    assert isinstance(wrapper.comms_matrix, ActiveCommunication), \
        "Communication matrix not initialized correctly"
    assert wrapper.failure_models == [probabilistic_model], \
        "Failure models not stored correctly"
    assert wrapper.agent_ids == list(test_env.possible_agents), \
        "Agent IDs not initialized correctly"


def test_initialization_error_no_models(test_env):
    """Test that initialization fails when no failure models are provided."""
    with pytest.raises(ValueError, match="No failure model"):
        CommunicationWrapper(env=test_env, failure_models=None)


def test_update_communication_state(test_env, probabilistic_model):
    """Test that the communication state is updated correctly."""
    wrapper = CommunicationWrapper(
        env=test_env,
        failure_models=[probabilistic_model]
    )

    # Set deterministic RNG
    wrapper.rng = np.random.RandomState(42)
    probabilistic_model.rng = wrapper.rng

    # Update communication state
    wrapper._update_communication_state()

    # Get the communication state
    comm_state = wrapper.get_communication_state()

    # With a failure probability of 0.5, approximately half of the links should be active
    # But with a fixed seed, the result should be deterministic
    # For each agent pair, check if the communication state matches expected
    # This depends on the specific RNG behavior with seed 42

    # Number of active links should be at least 1
    active_links = np.sum(comm_state)
    assert active_links > 0, "No active links in communication state"

    # Number of active links should be less than total possible links
    total_links = len(test_env.possible_agents) * (len(test_env.possible_agents) - 1)
    assert active_links < total_links, "All links are active (no failures)"


def test_filter_action(test_env, probabilistic_model):
    """Test that actions are filtered correctly based on communication state."""
    wrapper = CommunicationWrapper(
        env=test_env,
        failure_models=[probabilistic_model]
    )

    # Set a specific communication state where agent0 cannot communicate with anyone
    wrapper.comms_matrix.reset()
    for agent in test_env.possible_agents:
        for other in test_env.possible_agents:
            if agent != other:
                # All agents can communicate except agent0
                if agent != test_env.possible_agents[0]:
                    wrapper.comms_matrix.update(agent, other, True)
                else:
                    wrapper.comms_matrix.update(agent, other, False)

    # Test filtering an action for agent0
    agent0 = test_env.possible_agents[0]
    action = test_env.action_space(agent0).sample()
    filtered = wrapper.filter_action(agent0, action)

    # Since agent0 cannot communicate, the action should be replaced with a no-op
    # The exact no-op depends on the action space
    action_space = test_env.action_space(agent0)
    if isinstance(action_space, spaces.Discrete):
        assert filtered == 0, "Discrete action not filtered to no-op (0)"
    elif isinstance(action_space, spaces.MultiDiscrete):
        assert np.all(filtered == 0), "MultiDiscrete action not filtered to all zeros"
    elif isinstance(action_space, spaces.Box):
        assert np.all(filtered == 0), "Box action not filtered to all zeros"

    # Test filtering an action for another agent that can communicate
    agent1 = test_env.possible_agents[1]
    action = test_env.action_space(agent1).sample()
    filtered = wrapper.filter_action(agent1, action)

    # Since agent1 can communicate, the action should not be filtered
    assert np.array_equal(filtered, action), "Action filtered when it shouldn't be"


def test_apply_comm_mask(test_env, probabilistic_model):
    """Test that observations are masked correctly based on communication state."""
    wrapper = CommunicationWrapper(
        env=test_env,
        failure_models=[probabilistic_model]
    )

    # Create a sample observation dictionary
    obs = {
        agent: np.ones(5) * (i + 1) for i, agent in enumerate(test_env.possible_agents)
    }

    # Set a specific communication state where agent0 can only receive from agent1
    wrapper.comms_matrix.reset()
    agent0 = test_env.possible_agents[0]
    agent1 = test_env.possible_agents[1]
    agent2 = test_env.possible_agents[2]

    # agent1 can communicate with agent0, but agent2 cannot
    wrapper.comms_matrix.update(agent1, agent0, True)
    wrapper.comms_matrix.update(agent2, agent0, False)

    # Apply mask for agent0
    masked_obs = wrapper._apply_comm_mask(obs, agent0)

    # agent0 should receive its own observation unchanged
    assert np.array_equal(masked_obs[agent0], obs[agent0]), \
        "Agent's own observation changed"

    # agent0 should receive agent1's observation unchanged
    assert np.array_equal(masked_obs[agent1], obs[agent1]), \
        "Observation from communicating agent changed"

    # agent0 should receive zeroed observation from agent2
    assert np.array_equal(masked_obs[agent2], np.zeros_like(obs[agent2])), \
        "Observation from non-communicating agent not zeroed"


def test_step(test_env, probabilistic_model):
    """Test that step updates the environment and communication state."""
    wrapper = CommunicationWrapper(
        env=test_env,
        failure_models=[probabilistic_model]
    )
    wrapper.reset(seed=42)

    # Get the current agent
    agent = wrapper.agent_selection

    # Sample an action
    action = wrapper.action_space(agent).sample()

    # Spy on the environment's step and the wrapper's _update_communication_state
    original_step = wrapper.env.step
    original_update = wrapper._update_communication_state

    step_called = [False]
    update_called = [False]

    def mock_step(action):
        step_called[0] = True
        return original_step(action)

    def mock_update():
        update_called[0] = True
        return original_update()

    wrapper.env.step = mock_step
    wrapper._update_communication_state = mock_update

    # Call step on the wrapper
    wrapper.step(action)

    # Check that environment's step was called
    assert step_called[0], "Environment step not called"

    # Check that communication state was updated
    assert update_called[0], "Communication state not updated"

    # Restore original methods
    wrapper.env.step = original_step
    wrapper._update_communication_state = original_update


def test_reset(test_env, probabilistic_model):
    """Test that reset updates the environment and communication state."""
    wrapper = CommunicationWrapper(
        env=test_env,
        failure_models=[probabilistic_model]
    )

    # Spy on the environment's reset and the wrapper's _update_communication_state
    original_reset = wrapper.env.reset
    original_update = wrapper._update_communication_state

    reset_called = [False]
    update_called = [False]

    def mock_reset(seed=None, options=None):
        reset_called[0] = True
        return original_reset(seed=seed, options=options)

    def mock_update():
        update_called[0] = True
        return original_update()

    wrapper.env.reset = mock_reset
    wrapper._update_communication_state = mock_update

    # Call reset on the wrapper
    seed = 42
    wrapper.reset(seed=seed)

    # Check that environment's reset was called
    assert reset_called[0], "Environment reset not called"

    # Check that communication state was updated
    assert update_called[0], "Communication state not updated"

    # Check that RNG was updated
    assert wrapper.seed_val == seed, "Seed not updated"

    # Check that failure models' RNGs were updated
    for model in wrapper.failure_models:
        assert model.rng is wrapper.rng, "Failure model RNG not updated"

    # Restore original methods
    wrapper.env.reset = original_reset
    wrapper._update_communication_state = original_update


def test_observe(test_env, probabilistic_model):
    """Test that observe returns masked observations."""
    wrapper = CommunicationWrapper(
        env=test_env,
        failure_models=[probabilistic_model]
    )
    wrapper.reset(seed=42)

    # Get current agent
    agent = wrapper.agent_selection

    # Spy on the wrapper's _update_communication_state and _apply_comm_mask
    original_update = wrapper._update_communication_state
    original_mask = wrapper._apply_comm_mask

    update_called = [False]
    mask_called = [False]

    def mock_update():
        update_called[0] = True
        return original_update()

    def mock_mask(obs, receiver):
        mask_called[0] = True
        return original_mask(obs, receiver)

    wrapper._update_communication_state = mock_update
    wrapper._apply_comm_mask = mock_mask

    # Call observe on the wrapper
    wrapper.observe(agent)

    # Check that communication state was updated
    assert update_called[0], "Communication state not updated"

    # Check that observations were masked
    assert mask_called[0], "Observations not masked"

    # Restore original methods
    wrapper._update_communication_state = original_update
    wrapper._apply_comm_mask = original_mask


def test_last(test_env, probabilistic_model):
    """Test that last returns masked observations."""
    wrapper = CommunicationWrapper(
        env=test_env,
        failure_models=[probabilistic_model]
    )
    wrapper.reset(seed=42)

    # Spy on the wrapper's _update_communication_state and _apply_comm_mask
    original_update = wrapper._update_communication_state
    original_mask = wrapper._apply_comm_mask

    update_called = [False]
    mask_called = [False]

    def mock_update():
        update_called[0] = True
        return original_update()

    def mock_mask(obs, receiver):
        mask_called[0] = True
        return original_mask(obs, receiver)

    wrapper._update_communication_state = mock_update
    wrapper._apply_comm_mask = mock_mask

    # Call last on the wrapper with observe=True
    wrapper.last(observe=True)

    # Check that communication state was updated
    assert update_called[0], "Communication state not updated"

    # Check that observations were masked
    assert mask_called[0], "Observations not masked"

    # Reset spy flags
    update_called[0] = False
    mask_called[0] = False

    # Call last with observe=False
    wrapper.last(observe=False)

    # Communication state should still be updated
    assert update_called[0], "Communication state not updated with observe=False"

    # But observations should not be masked
    assert not mask_called[0], "Observations masked with observe=False"

    # Restore original methods
    wrapper._update_communication_state = original_update
    wrapper._apply_comm_mask = original_mask
    obs, *_ = wrapper.last(observe=True)
    print("Returned obs type:", type(obs))
    print("Observation:", obs)

    print("Agent selected:", wrapper.env.agent_selection)
    print("Raw obs from env.last:", obs)


def test_add_failure_model(test_env, probabilistic_model, distance_model):
    """Test that add_failure_model correctly adds a new model."""
    wrapper = CommunicationWrapper(
        env=test_env,
        failure_models=[probabilistic_model]
    )

    # Initially only has the probabilistic model
    assert len(wrapper.failure_models) == 1, "Initial model count incorrect"
    assert wrapper.failure_models[0] == probabilistic_model, "Initial model incorrect"

    # Add the distance model
    wrapper.add_failure_model(distance_model)

    # Now should have both models
    assert len(wrapper.failure_models) == 2, "Model count after adding incorrect"
    assert wrapper.failure_models[0] == probabilistic_model, "First model changed"
    assert wrapper.failure_models[1] == distance_model, "Second model not added correctly"

    # The added model's RNG should be set to the wrapper's RNG
    assert distance_model.rng is wrapper.rng, "Added model's RNG not set correctly"


def test_multiple_failure_models(test_env, probabilistic_model, distance_model):
    """Test that multiple failure models are applied in sequence."""
    wrapper = CommunicationWrapper(
        env=test_env,
        failure_models=[probabilistic_model, distance_model]
    )
    wrapper.reset(seed=42)

    # Both models should affect the communication matrix
    # The distance model will make nearby agents communicate, but the probabilistic
    # model will add random failures

    # Update communication state
    wrapper._update_communication_state()

    # Get the communication state
    comm_state = wrapper.get_communication_state()

    # Agent 0 and 1 are close enough to communicate per the distance model,
    # but the probabilistic model might block some links
    # Agents 0 and 2 are too far apart according to the distance model,
    # so they should never communicate

    # Agent 0 and 2 should never communicate
    agent0_idx = test_env.possible_agents.index(test_env.possible_agents[0])
    agent2_idx = test_env.possible_agents.index(test_env.possible_agents[2])

    assert not comm_state[agent0_idx, agent2_idx], "Agent 0 and 2 are communicating when they shouldn't"
    assert not comm_state[agent2_idx, agent0_idx], "Agent 2 and 0 are communicating when they shouldn't"


def test_comm_masking_logic():
    # Step 1: Define agent IDs and dummy observations
    agent_ids = ["agent_0", "agent_1", "agent_2"]
    dummy_obs = {
        "agent_0": np.ones(5),
        "agent_1": np.full(5, 2),
        "agent_2": np.full(5, 3),
    }

    # Step 2: Define a known communication matrix (sender → receiver)
    # comm[i][j] == 1 means sender i is visible to receiver j
    comm_matrix = np.array([
        [1, 1, 0],  # agent_0 can be seen by 0 and 1
        [1, 1, 0],  # agent_1 can be seen by 0 and 1
        [1, 1, 1],  # agent_2 can be seen by all
    ])

    # Step 3: Mock a CommunicationWrapper with a fake env and fixed comm state
    class DummyEnv:
        def __init__(self):

            self.possible_agents = ["agent_0", "agent_1", "agent_2"]
            self.agents = self.possible_agents.copy()
        def observe(self, agent): return dummy_obs.copy()
        def reset(self): pass

    wrapper = CommunicationWrapper(env=DummyEnv(), failure_models=[])
    wrapper.agent_ids = agent_ids
    wrapper.comms_matrix = ActiveCommunication(agent_ids)
    wrapper.comms_matrix.matrix = comm_matrix

    # Step 4: Apply masking for each agent
    for receiver in agent_ids:
        obs = dummy_obs.copy()
        masked = wrapper._apply_comm_mask(obs.copy(), receiver)

        for sender in agent_ids:
            if sender == receiver:
                continue

            allowed = comm_matrix[agent_ids.index(sender), agent_ids.index(receiver)]
            is_masked = np.allclose(masked[sender], 0)

            if allowed:
                assert not is_masked, f"❌ {receiver} should see {sender}, but it was masked!"
            else:
                assert is_masked, f"❌ {receiver} should NOT see {sender}, but it was visible!"

    print("✅ All masking logic assertions passed.")