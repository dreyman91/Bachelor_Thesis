import pytest
import numpy as np
from pettingzoo.utils.env import AECEnv
from gymnasium import spaces
from Failure_API.src.failure_api.wrappers.communication_wrapper import CommunicationWrapper
from Failure_API.src.failure_api.communication_models.probabilistic_model import ProbabilisticModel


class MockEnv(AECEnv):
    """A minimal mock environment for testing the communication wrapper"""

    def __init__(self, num_agents=3):
        self.possible_agents = [f"agent_{i}" for i in range(num_agents)]
        self.agents = self.possible_agents[:]
        self.agent_selection = self.agents[0]
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.observation_spaces = {agent: spaces.Box(low=0, high=1, shape=(5,)) for agent in self.agents}
        self.action_spaces = {agent: spaces.Discrete(5) for agent in self.agents}
        self._observations = {agent: {a: np.ones(5) * (i + 1) for i, a in enumerate(self.agents)}
                              for agent in self.agents}

    def observe(self, agent):
        """Return a dictionary mapping each agent to its observation"""
        return self._observations[agent]

    def step(self, action):
        """Move to the next agent in the cycle"""
        idx = self.agents.index(self.agent_selection)
        self.agent_selection = self.agents[(idx + 1) % len(self.agents)]

    def reset(self, seed=None, options=None):
        """Reset the environment state"""
        self.agents = self.possible_agents[:]
        self.agent_selection = self.agents[0]
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        return None

    def action_space(self, agent):
        return self.action_spaces[agent]

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def last(self, observe=True):
        agent = self.agent_selection
        obs = self.observe(agent) if observe else None
        return obs, 0.0, False, False, {}


class TestCommunicationWrapper:
    """Comprehensive tests for the CommunicationWrapper's masking logic"""

    @pytest.fixture
    def env(self):
        """Create a fresh environment for each test"""
        return MockEnv()

    def test_perfect_communication(self, env):
        """Test with p=0 (perfect communication)"""
        agent_ids = env.possible_agents
        failure_model = ProbabilisticModel(agent_ids=agent_ids, failure_prob=0.0)
        wrapped_env = CommunicationWrapper(env, failure_models=[failure_model])

        # Reset environment to initialize communication state
        wrapped_env.reset(seed=42)  # Use fixed seed for reproducibility

        # Verify all agents can see each other (except themselves)
        for agent_i in agent_ids:
            # Get observation for agent_i
            obs_i = wrapped_env.observe(agent_i)

            # Check each agent's observations
            for agent_j in agent_ids:
                # Agents should see all others but not themselves in the matrix
                if agent_i != agent_j:
                    # Verify agent_j is in agent_i's observation
                    assert agent_j in obs_i, f"Agent {agent_i} should see {agent_j} with p=0"

                    # Verify the observation content is preserved
                    expected_obs = env._observations[agent_i][agent_j]
                    np.testing.assert_array_equal(obs_i[agent_j], expected_obs)

        # Verify communication matrix matches our expectations
        comm_matrix = wrapped_env.get_communication_state().astype(int)
        expected_matrix = np.ones((3, 3), dtype=int)
        np.fill_diagonal(expected_matrix, 0)  # No self-communication
        np.testing.assert_array_equal(comm_matrix, expected_matrix)

    def test_no_communication(self, env):
        """Test with p=1 (no communication)"""
        agent_ids = env.possible_agents
        failure_model = ProbabilisticModel(agent_ids=agent_ids, failure_prob=1.0)
        wrapped_env = CommunicationWrapper(env, failure_models=[failure_model])

        wrapped_env.reset(seed=42)

        # Verify no agents can see each other
        for agent_i in agent_ids:
            obs_i = wrapped_env.observe(agent_i)

            # Check observation dictionary is empty or contains only self
            assert len(obs_i) <= 1, f"Agent {agent_i} should see at most itself with p=1, saw {obs_i.keys()}"

            for agent_j in agent_ids:
                if agent_i != agent_j:
                    # Verify agent_j is NOT in agent_i's observation
                    assert agent_j not in obs_i, f"Agent {agent_i} should NOT see {agent_j} with p=1"

        # Verify communication matrix is all zeros (except diagonal)
        comm_matrix = wrapped_env.get_communication_state().astype(int)
        expected_matrix = np.zeros((3, 3), dtype=int)
        np.testing.assert_array_equal(comm_matrix, expected_matrix)

    def test_probabilistic_communication(self, env):
        """Test with p=0.5 (probabilistic communication)"""
        agent_ids = env.possible_agents
        failure_model = ProbabilisticModel(agent_ids=agent_ids, failure_prob=0.5)
        wrapped_env = CommunicationWrapper(env, failure_models=[failure_model])

        # Run multiple resets to verify probabilistic behavior
        results = []

        for seed in range(10):  # Try 10 different seeds
            wrapped_env.reset(seed=seed)

            # Get communication matrix
            comm_matrix = wrapped_env.get_communication_state().astype(int)

            # For each agent pair, check if observations match matrix state
            for i, agent_i in enumerate(agent_ids):
                obs_i = wrapped_env.observe(agent_i)

                for j, agent_j in enumerate(agent_ids):
                    if agent_i == agent_j:
                        continue

                    # Matrix state determines if agent_j can communicate with agent_i
                    should_see = comm_matrix[j][i] == 1
                    actually_sees = agent_j in obs_i

                    # Record if observation matches matrix state
                    results.append(should_see == actually_sees)

        # Assert all observations match the matrix state
        assert all(results), "Observations should always match communication matrix state"

        # Assert we have a mix of successful and failed communications
        # This verifies the probabilistic nature is working
        comm_matrices = []
        for seed in range(50):  # Generate many matrices to verify randomness
            wrapped_env.reset(seed=seed)
            comm_matrices.append(wrapped_env.get_communication_state().astype(int))

        # Convert to a single array and get non-diagonal elements
        all_comm = np.array(comm_matrices)
        non_diag_mask = ~np.eye(len(agent_ids), dtype=bool)
        non_diag_values = all_comm[:, non_diag_mask]

        # Check we have a mix of 0s and 1s (some communication succeeded, some failed)
        zeros = np.sum(non_diag_values == 0)
        ones = np.sum(non_diag_values == 1)

        # With p=0.5, we expect roughly equal zeros and ones
        ratio = ones / (zeros + ones)
        assert 0.4 <= ratio <= 0.6, f"Expected ~0.5 success rate, got {ratio}"

    def test_consistency_within_step(self, env):
        """Test that masking is consistent within a step"""
        agent_ids = env.possible_agents
        failure_model = ProbabilisticModel(agent_ids=agent_ids, failure_prob=0.5)
        wrapped_env = CommunicationWrapper(env, failure_models=[failure_model])

        wrapped_env.reset(seed=42)

        # Get matrix state
        comm_matrix = wrapped_env.get_communication_state().astype(int)

        # Check multiple observations within same step are consistent
        for agent_i in agent_ids:
            # Get observation multiple times
            obs1 = wrapped_env.observe(agent_i)
            obs2 = wrapped_env.observe(agent_i)
            obs3 = wrapped_env.observe(agent_i)

            # Verify all observations are identical
            assert set(obs1.keys()) == set(obs2.keys()) == set(obs3.keys()), \
                "Observations within the same step should be consistent"

            # Verify all observations match the matrix state
            for j, agent_j in enumerate(agent_ids):
                if agent_i == agent_j:
                    continue

                # Get matrix entry
                should_see = comm_matrix[j][i] == 1

                # Check observation matches matrix
                actually_sees = agent_j in obs1
                assert should_see == actually_sees, \
                    f"Observation doesn't match matrix: agent {agent_i} seeing {agent_j}"

    def test_matrix_changes_between_steps(self, env):
        """Test that the matrix changes between steps with probabilistic model"""
        agent_ids = env.possible_agents
        failure_model = ProbabilisticModel(agent_ids=agent_ids, failure_prob=0.5)
        wrapped_env = CommunicationWrapper(env, failure_models=[failure_model])

        wrapped_env.reset(seed=42)

        # Store initial matrix
        initial_matrix = wrapped_env.get_communication_state().copy()

        # Take a step and check if matrix changed
        wrapped_env.step(0)  # Action doesn't matter for this test

        # Get new matrix
        new_matrix = wrapped_env.get_communication_state().copy()

        # Matrices should be different with p=0.5
        assert not np.array_equal(initial_matrix, new_matrix), \
            "Communication matrix should change between steps with probabilistic model"

    def test_action_filtering(self, env):
        """Test that actions are filtered based on communication state"""
        agent_ids = env.possible_agents
        failure_model = ProbabilisticModel(agent_ids=agent_ids, failure_prob=1.0)  # No communication
        wrapped_env = CommunicationWrapper(env, failure_models=[failure_model])

        wrapped_env.reset(seed=42)

        # With p=1.0, no agent can communicate, so all actions should be filtered to no-op
        for agent in agent_ids:
            # Try a non-zero action
            action = 3
            filtered_action = wrapped_env.filter_action(agent, action)

            # Since no communication is possible, should return no-op (0)
            assert filtered_action == 0, \
                f"Action should be filtered to no-op with p=1.0, got {filtered_action}"

    def test_step_counter_increments(self, env):
        """Test that step counter increments correctly"""
        agent_ids = env.possible_agents
        failure_model = ProbabilisticModel(agent_ids=agent_ids, failure_prob=0.5)
        wrapped_env = CommunicationWrapper(env, failure_models=[failure_model])

        wrapped_env.reset(seed=42)

        # Initial step counter should be 0
        assert wrapped_env._step_counter == 0

        # Take steps and verify counter increments
        for i in range(1, 10):
            wrapped_env.step(0)
            assert wrapped_env._step_counter == i, \
                f"Step counter should be {i}, got {wrapped_env._step_counter}"