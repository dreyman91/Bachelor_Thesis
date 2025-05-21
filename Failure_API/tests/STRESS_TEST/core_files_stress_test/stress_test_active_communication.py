from mpe2 import simple_spread_v3
from Failure_API.src.failure_api.communication_models.active_communication import ActiveCommunication
from pettingzoo.utils.conversions import aec_to_parallel
import pytest
import numpy as np
from scipy.sparse import lil_matrix

# ============= PettingZoo===============#
def test_active_comm_with_aec_env():
    env = simple_spread_v3.env(N=3, max_cycles=5)
    env.reset(seed=42)
    agents = env.possible_agents

    comm = ActiveCommunication(agent_ids=agents)

    for step in range(5):
        current_agent = env.agent_selection
        obs, rew, term, trunc, info = env.last()
        act = env.action_space(current_agent).sample()
        env.step(act)

        # Update communication
        for i in range(len(agents)):
            for j in range(len(agents)):
                if i != j:
                    comm.update(agents[i], agents[j], 0.5 if step % 2 == 0 else 0.0)

        matrix = comm.get_boolean_matrix()
        assert matrix.shape == (len(agents), len(agents))
        assert (matrix.diagonal() == 0).all()

def test_active_comm_with_parallel_env_stress():
    N = 50
    env = simple_spread_v3.env(N=N, max_cycles=10)
    env.reset(seed=123)
    parallel_env = aec_to_parallel(env)

    agent_ids = parallel_env.possible_agents
    comm = ActiveCommunication(agent_ids=agent_ids)

    for step in range(10):
        actions = {agent: parallel_env.action_space(agent).sample() for agent in parallel_env.agents}
        parallel_env.step(actions)

        for sender in agent_ids:
            for receiver in agent_ids:
                if sender != receiver:
                    comm.update(sender, receiver, 1.0 if step % 3 else 0.0)

        # Verify no crash and correct matrix size
        assert comm.get_boolean_matrix().shape == (N, N)

# =================== PettingZoo ===================#

def test_reset_with_new_agents():
    ac = ActiveCommunication(["a1", "a2", "a3"])
    ac.update("a1", "a2", 0.5)
    ac.reset(agent_ids=["b1", "b2"])
    assert set(ac.agent_ids) == {"b1", "b2"}
    assert ac.get("b1", "b2") is True


def test_non_float_initial_matrix():
    bad_matrix = np.array([[1, 1], [1, 1]], dtype=int)
    with pytest.raises(ValueError, match="float type"):
        ActiveCommunication(["x", "y"], initial_matrix=bad_matrix)


# ========= Initialization & Validation Tests ===========#

def test_default_matrix_shape_and_values():
    ac = ActiveCommunication(agent_ids=["a1", "a2", "a3"])
    matrix = ac.get_state().toarray()
    assert matrix.shape == (3, 3)
    assert np.all(matrix.diagonal() == 0.0)
    assert np.all(matrix + np.eye(3) > 0.0)

def test_invalid_matrix_shape():
    bad_matrix = np.ones((2, 2))
    with pytest.raises(ValueError):
        ActiveCommunication(agent_ids=["a1", "a2", "a3"], initial_matrix=bad_matrix)


# ========= Update & Bandwidth Logic ===========#

def test_update_and_get_bandwidth():
    ac = ActiveCommunication(["x", "y"])
    ac.update("x", "y", 0.75)
    assert ac.get_bandwidth("x", "y") == 0.75
    assert ac.can_communicate("x", "y")

def test_boolean_bandwidth_conversion():
    ac = ActiveCommunication(["a", "b"])
    ac.update("a", "b", False)
    assert ac.get("a", "b") is False
    ac.update("a", "b", True)
    assert ac.get("a", "b") is True


# =========  Masking & Matrix Logic ===========#

def test_boolean_matrix_thresholding():
    ac = ActiveCommunication(["u", "v"])
    ac.update("u", "v", 0.4)
    matrix = ac.get_boolean_matrix(threshold=0.5)
    assert matrix[0, 1] is False
    matrix = ac.get_boolean_matrix(threshold=0.3)
    assert matrix[0, 1] is True

def test_blocked_agents_logic():
    ac = ActiveCommunication(["x", "y", "z"])
    ac.update("x", "y", 0.0)
    blocked = ac.get_blocked_agents("x")
    assert "y" in blocked
    assert "z" not in blocked

# =========  Matrix Integrity & Behavior ===========#

def test_matrix_copy_behavior():
    ac = ActiveCommunication(["a", "b"])
    matrix_1 = ac.get_state()
    matrix_1[0, 1] = 999
    matrix_2 = ac.get_state()
    assert matrix_2[0, 1] != 999

def test_reset_communication_matrix():
    ac = ActiveCommunication(["i", "j"])
    ac.update("i", "j", 0.2)
    ac.reset()
    assert ac.get("i", "j") is True


# =========  Edge & Stress Scenarios ===========#

def test_large_matrix_stress():
    agent_ids = [f"agent_{i}" for i in range(100)]
    ac = ActiveCommunication(agent_ids)
    for i in range(100):
        for j in range(100):
            if i != j:
                ac.update(agent_ids[i], agent_ids[j], 0.1)
    matrix = ac.get_boolean_matrix()
    assert matrix.shape == (100, 100)

def test_get_with_unknown_agent():
    ac = ActiveCommunication(["a", "b"])
    with pytest.raises(KeyError):
        ac.get("a", "non_existent")

