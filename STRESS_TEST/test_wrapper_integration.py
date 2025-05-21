# test_wrapper_integration.py

from pettingzoo.utils.conversions import aec_to_parallel
import pytest
import numpy as np
from mpe2 import simple_spread_v3
from pettingzoo.utils.conversions import aec_to_parallel
from Failure_API.src.failure_api.wrappers.communication_wrapper import CommunicationWrapper
from Failure_API.src.failure_api.wrappers import SharedObsWrapper, BaseWrapper
from Failure_API.src.failure_api.wrappers.noise_wrapper import NoiseWrapper
from Failure_API.src.failure_api.communication_models import (ProbabilisticModel,
                                                              ActiveCommunication, BaseMarkovModel, DistanceModel)
from Failure_API.src.failure_api.noise_models.gaussian_noise import GaussianNoiseModel


# @pytest.mark.parametrize("wrapper_stack", [
#     ["comm"],
#     ["comm", "sharedobs"],
#     ["comm", "noise"],
#     ["comm", "sharedobs", "noise"]
# ])
# def test_wrapper_pipeline_runs(wrapper_stack):
#     N = 3
#     env = simple_spread_v3.env(N=N, max_cycles=10)
#     env.reset(seed=42)
#
#     # Communication layer
#     if "comm" in wrapper_stack:
#         model = ProbabilisticModel(env.possible_agents, failure_prob=0.5)
#         env = CommunicationWrapper(env, failure_models=[model])
#
#     # Shared Observation
#     if "sharedobs" in wrapper_stack:
#         env = SharedObsWrapper(env)
#
#     # Noise Model
#     if "noise" in wrapper_stack:
#         noise_model = GaussianNoiseModel(std=0.1)
#         env = NoiseWrapper(env, noise_model=noise_model)
#
#     env.reset(seed=42)
#     for _ in range(5):
#         for agent in env.agent_iter():
#             obs, reward, termination, truncation, info = env.last()
#             if termination or truncation:
#                 action = None
#             else:
#                 action = env.action_space(agent).sample()
#             env.step(action)
#
#         obs = {agent: env.observe(agent) for agent in env.agents}
#         assert isinstance(obs, dict)
#         for agent in env.agents:
#             assert obs[agent] is not None
#
#
#
#
# def test_wrapper_stack_in_parallel_mode():
#     N = 3
#     base_env = simple_spread_v3.env(N=N, max_cycles=10)
#     base_env.reset(seed=42)
#
#     model = ProbabilisticModel(base_env.possible_agents, failure_prob=0.5)
#     env = CommunicationWrapper(base_env, failure_models=[model])
#     env = SharedObsWrapper(env)
#     env = NoiseWrapper(env, noise_model=GaussianNoiseModel(std=0.1))
#
#     parallel_env = aec_to_parallel(env)
#     obs, _ = parallel_env.reset(seed=42)
#
#     for _ in range(5):
#         actions = {agent: parallel_env.action_space(agent).sample() for agent in parallel_env.agents}
#         obs, rewards, terminations, truncations, infos = parallel_env.step(actions)
#
#         assert isinstance(obs, dict)
#         for agent in parallel_env.agents:
#             if not terminations[agent] and not truncations[agent]:
#                 assert agent in obs
#                 assert obs[agent] is not None, f"{agent} returned None observation unexpectedly"
#
#
#
# def test_sharedobs_comm_noise_chain():
#     env = simple_spread_v3.env(N=3, max_cycles=25)
#     env.reset(seed=0)
#     comm_model = ProbabilisticModel(env.possible_agents, failure_prob=0.3)
#     env = CommunicationWrapper(env, failure_models=[comm_model])
#     env = SharedObsWrapper(env)
#     env = NoiseWrapper(env, noise_model=GaussianNoiseModel(std=0.1))
#     env.reset(seed=0)
#     for _ in range(5):
#         actions = {a: env.action_space(a).sample() for a in env.agents}
#         for agent in env.agent_iter():
#             obs, reward, termination, truncation, info = env.last()
#             if termination or truncation:
#                 env.step(None)
#             else:
#                 action = env.action_space(agent).sample()
#                 env.step(action)
#         obs = {a: env.observe(a) for a in env.agents}
#         for a in env.agents:
#             assert isinstance(obs[a], np.ndarray)
#             assert not np.isnan(obs[a]).any()
#
#
# def test_wrapper_reset_consistency():
#     env = simple_spread_v3.env(N=3, max_cycles=10)
#     model = ProbabilisticModel(env.possible_agents, failure_prob=0.5)
#     env = CommunicationWrapper(env, failure_models=[model])
#     env = SharedObsWrapper(env)
#
#     env.reset(seed=42)
#     obs1 = {a: env.observe(a) for a in env.agents}
#
#     env.reset(seed=42)
#     obs2 = {a: env.observe(a) for a in env.agents}
#
#     for a in env.agents:
#         if obs1[a] is None or obs2[a] is None:
#             continue  # skip agents with dropped communication
#         assert isinstance(obs1[a], dict)
#         assert isinstance(obs2[a], dict)
#         for k in obs1[a]:
#             assert k in obs2[a]
#             assert isinstance(obs1[a][k], np.ndarray)
#             assert isinstance(obs2[a][k], np.ndarray)
#             assert obs1[a][k].shape == obs2[a][k].shape
#
#
#
#
# def test_wrapper_combination_masked_observation_keys():
#     env = simple_spread_v3.env(N=3, max_cycles=25)
#     model = ProbabilisticModel(env.possible_agents, failure_prob=0.5)
#     env = CommunicationWrapper(env, failure_models=[model])
#     env = SharedObsWrapper(env)
#     env.reset(seed=123)
#
#     for _ in range(3 * len(env.possible_agents)):
#         agent = env.agent_selection
#         obs, reward, term, trunc, info = env.last()
#         if term or trunc:
#             env.step(None)
#         else:
#             env.step(env.action_space(agent).sample())
#
#     for a in env.agents:
#         ob = env.observe(a)
#         if ob is None:
#             continue  # agent missed communication
#         assert isinstance(ob, dict)
#         for k, v in ob.items():
#             assert isinstance(v, np.ndarray)
#             assert not np.isnan(v).any()
#
#
# def test_invalid_markov_matrix_shape():
#     with pytest.raises(ValueError):
#         BaseMarkovModel(["a", "b"], default_matrix=np.array([[1.0]]))
#
#
def test_invalid_markov_probabilities():
    with pytest.raises(ValueError):
        BaseMarkovModel(["a", "b"], default_matrix=np.array([[0.8, 0.8], [0.2, 0.2]]))


def test_nan_positions_distance_model():
    def nan_pos_fn(agent=None):
        return {"a": np.array([np.nan, 0]), "b": np.array([0, np.nan])} if agent is None else {"a": np.array([np.nan, 0]), "b": np.array([0, np.nan])}[agent]

    matrix = ActiveCommunication(["a", "b"])
    model = DistanceModel(["a", "b"], distance_threshold=10.0, pos_fn=nan_pos_fn)
    with pytest.raises(ValueError):
        model.update_connectivity(matrix)


def test_noise_model_requires_rng():
    model = GaussianNoiseModel(std=1.0)
    obs = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="Random number generator not initialized"):
        model.apply(obs)

def test_observe_with_missing_agent_key():
    env = simple_spread_v3.env(N=3, max_cycles=5)
    env = CommunicationWrapper(env, failure_models=[])
    env = SharedObsWrapper(env)
    env = NoiseWrapper(env, noise_model=GaussianNoiseModel(std=0.1))
    env.reset(seed=42)

    missing_agent = "nonexistent_agent"
    with pytest.raises(KeyError):
        _ = env.observe(missing_agent)

def test_noise_model_rng_injection_from_wrapper():
    env = simple_spread_v3.env(N=3, max_cycles=5)

    # Create the model manually
    model = GaussianNoiseModel(std=0.1)
    assert model.rng is None  # Ensure uninitialized

    # Wrap the env with the model
    env = NoiseWrapper(env, noise_model=model)
    env.reset(seed=42)

    # RNG should now be injected by wrapper
    assert model.rng is not None
    assert isinstance(model.rng, np.random.Generator)

    # Confirm it can apply noise without error
    obs = env.observe("agent_0")
    assert isinstance(obs, np.ndarray)

#
# def test_noise_wrapper_inherits_base_wrapper():
#     env = simple_spread_v3.env(N=3, max_cycles=5)
#     model = GaussianNoiseModel(std=0.1)
#     wrapper = NoiseWrapper(env, noise_model=model)
#
#     # Check class inheritance
#     assert isinstance(wrapper, BaseWrapper), "NoiseWrapper should inherit from BaseWrapper"
#
#     # Check reset_rng is present
#     assert hasattr(wrapper, "reset_rng"), "NoiseWrapper should inherit reset_rng from BaseWrapper"
#
#     # Test that reset_rng sets model RNG
#     wrapper.reset_rng(seed=123)
#     assert model.rng is not None
#     assert isinstance(model.rng, np.random.Generator)
