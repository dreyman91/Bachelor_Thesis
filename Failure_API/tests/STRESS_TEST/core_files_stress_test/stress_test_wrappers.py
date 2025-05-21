from Failure_API.src.failure_api.wrappers import BaseWrapper, SharedObsWrapper, CommunicationWrapper, NoiseWrapper
from Failure_API.src.failure_api.communication_models import (ProbabilisticModel, DistanceModel, BaseMarkovModel,
                                                              SignalBasedModel, DelayBasedModel)
from Failure_API.src.failure_api.noise_models import LaplacianNoiseModel, GaussianNoiseModel

from mpe2 import simple_spread_v3

# ====== T18: Observation Format Is Dict of Arrays ======#
def test_observation_is_dict_of_arrays():
    from pettingzoo.mpe import simple_spread_v3
    from Failure_API.src.failure_api.wrappers.communication_wrapper import CommunicationWrapper
    from Failure_API.src.failure_api.communication_models.probabilistic_model import ProbabilisticModel
    import numpy as np

    env = simple_spread_v3.env(N=4)
    model = ProbabilisticModel(agent_ids=env.possible_agents, failure_prob=0.3)
    env = CommunicationWrapper(env, failure_models=[model])
    env.reset(seed=99)

    for agent in env.agents:
        obs = env.observe(agent)
        assert isinstance(obs, dict), f"Expected dict for observation, got {type(obs)}"
        for k, v in obs.items():
            assert isinstance(v, np.ndarray), f"Expected numpy array for obs[{k}], got {type(v)}"

#
# # ====== T01: BaseWrapper RNG reset ======#
# def test_rng_reset_consistency():
#     env = simple_spread_v3.env()
#     wrapper = BaseWrapper(env, seed=123)
#     rng_value_1 = wrapper.rng.integers(0, 1000)
#     wrapper.reset_rng(seed=123)
#     rng_value_2 = wrapper.rng.integers(0, 1000)
#     assert rng_value_1 == rng_value_2, "RNG reset did not produce consistent values"
#
#
# # ====== T02: SharedObsWrapper caching ======#
# def test_shared_obswrapper_caching():
#     env = SharedObsWrapper(simple_spread_v3.env(N=3))
#     env.reset(seed=42)
#     obs1 = env.observe_all()
#     obs2 = env.observe_all()
#     assert obs1 == obs2, "Observations not cached correctly between repeated calls"
#
#
# # ====== T03: CommunicationWrapper filter_action ======#
# def test_filter_action_returns_noop_if_isolated():
#     env = simple_spread_v3.env(N=3)
#     model = ProbabilisticModel(agent_ids=env.possible_agents, failure_prob=1.0)
#     wrapper = CommunicationWrapper(env, failure_models=[model])
#     wrapper.reset(seed=42)
#     agent = wrapper.env.agent_selection
#     action = wrapper.env.action_space(agent).sample()
#     filtered = wrapper.filter_action(agent, action)
#     assert filtered in [0, wrapper.env.action_space(agent).no_op_index], "No-op not returned on total comms fail"
#
#
# # ====== T04: CommunicationWrapper step terminated agent ======#
# def test_step_with_terminated_agent():
#     env = simple_spread_v3.env(N=2, max_cycles=1)
#     model = ProbabilisticModel(agent_ids=env.possible_agents, failure_prob=0.5)
#     wrapper = CommunicationWrapper(env, failure_models=[model])
#     wrapper.reset(seed=1)
#     for _ in range(10):
#         agent = wrapper.env.agent_selection
#         if agent not in wrapper.env.agents:
#             wrapper.env.step(None)
#         else:
#             act = wrapper.env.action_space(agent).sample()
#             wrapper.step(act)
#     assert True  # If no crash, pass
#
#
# # ====== T05: CommunicationWrapper masking logic ======#
# def test_comm_mask_shapes_and_zeros():
#     def dummy_pos_fn(agent):
#         return np.zeros(2)
#
#     model = DistanceModel(agent_ids=env.possible_agents, distance_threshold=0.01, pos_fn=dummy_pos_fn)
#     wrapper = CommunicationWrapper(env, failure_models=[model])
#     wrapper.reset(seed=2)
#     agent = wrapper.env.agent_selection
#     obs = wrapper.observe(agent)
#     for sender, o in obs.items():
#         if sender != agent:
#             assert np.all(o == 0), f"Expected masked obs for {sender}, got non-zero"
#
#
# # ====== T06: Add model dynamically ======#
# def test_add_failure_model_dynamically():
#     env = simple_spread_v3.env(N=3)
#     wrapper = CommunicationWrapper(env, failure_models=[], agent_ids=env.possible_agents)
#     model = ProbabilisticModel(agent_ids=env.possible_agents, failure_prob=0.2)
#     wrapper.add_failure_model(model)
#     assert len(wrapper.failure_models) == 1
#
#
# # ====== T07: Observe fails gracefully on agent exit ======#
# def test_observe_returns_zeros_for_terminated():
#     from Failure_API.src.failure_api.wrappers.communication_wrapper import CommunicationWrapper
#     from Failure_API.src.failure_api.communication_models.probabilistic_model import ProbabilisticModel
#     import numpy as np
#
#     env = simple_spread_v3.env(N=2, max_cycles=1)
#     model = ProbabilisticModel(agent_ids=env.possible_agents, failure_prob=1.0)
#     wrapper = CommunicationWrapper(env, failure_models=[model])
#     wrapper.reset(seed=3)
#     for _ in range(10):
#         if wrapper.env.agents:
#             a = wrapper.env.agent_selection
#             wrapper.step(wrapper.env.action_space(a).sample())
#     for a in env.possible_agents:
#         result = wrapper.observe(a)
#         if isinstance(result, dict):
#             assert all(np.all(v == 0) for v in result.values())
#         else:
#             assert np.all(result == 0)
#
#
# # ====== T08: SharedObsWrapper compatibility ======#
# def test_sharedobswrapper_with_step_observe_last():
#     from Failure_API.src.failure_api.wrappers.sharedobs_wrapper import SharedObsWrapper
#     env = SharedObsWrapper(simple_spread_v3.env(N=3))
#     env.reset(seed=42)
#     for _ in range(5):
#         if env.agents:
#             agent = env.agent_selection
#             action = env.action_space(agent).sample()
#             env.step(action)
#             obs = env.observe(agent)
#             assert obs is not None
#             _ = env.last(observe=True)
#
#
# # ====== T09: CommunicationWrapper + NoiseWrapper ======#
# def test_nested_communication_noise_wrappers():
#     from Failure_API.src.failure_api.wrappers.communication_wrapper import CommunicationWrapper
#     from Failure_API.src.failure_api.wrappers.noise_wrapper import NoiseWrapper
#     from Failure_API.src.failure_api.noise_models.gaussian_noise import GaussianNoiseModel
#     from Failure_API.src.failure_api.communication_models.probabilistic_model import ProbabilisticModel
#     env = simple_spread_v3.env(N=3)
#     comm_model = ProbabilisticModel(agent_ids=env.possible_agents, failure_prob=0.5)
#     env = CommunicationWrapper(env, failure_models=[comm_model])
#     noise_model = GaussianNoiseModel(std=0.1)
#     env = NoiseWrapper(env, noise_model=noise_model)
#     env.reset(seed=123)
#     agent = env.agent_selection
#     obs = env.observe(agent)
#     assert obs is not None
#
#
# # ====== T10: AEC mode test ======#
# def test_aec_mode_full_cycle():
#     from Failure_API.src.failure_api.wrappers.communication_wrapper import CommunicationWrapper
#     from Failure_API.src.failure_api.communication_models.probabilistic_model import ProbabilisticModel
#     env = simple_spread_v3.env(N=3, max_cycles=10)
#     model = ProbabilisticModel(agent_ids=env.possible_agents, failure_prob=0.2)
#     env = CommunicationWrapper(env, failure_models=[model])
#     env.reset(seed=10)
#     for _ in range(30):
#         if env.agents:
#             agent = env.agent_selection
#             action = env.action_space(agent).sample()
#             env.step(action)
#     assert True
#
#
# # ====== T11: ParallelEnv PettingZoo ======#
# def test_parallel_env_conversion():
#     from Failure_API.src.failure_api.wrappers.communication_wrapper import CommunicationWrapper
#     from Failure_API.src.failure_api.communication_models.probabilistic_model import ProbabilisticModel
#     from pettingzoo.utils.conversions import aec_to_parallel
#     env = simple_spread_v3.env(N=50)
#     model = ProbabilisticModel(agent_ids=env.possible_agents, failure_prob=0.4)
#     env = CommunicationWrapper(env, failure_models=[model])
#     env = aec_to_parallel(env)
#     obs, _ = env.reset(seed=42)
#     assert isinstance(obs, dict)
#     assert all(agent in obs for agent in env.agents)
#
#
# # ====== T12: PettingZoo reset/step/full loop ======#
# def test_full_reset_step_observe_cycle():
#     from Failure_API.src.failure_api.wrappers.communication_wrapper import CommunicationWrapper
#     from Failure_API.src.failure_api.communication_models.probabilistic_model import ProbabilisticModel
#     env = simple_spread_v3.env(N=4)
#     model = ProbabilisticModel(agent_ids=env.possible_agents, failure_prob=0.3)
#     env = CommunicationWrapper(env, failure_models=[model])
#     for episode in range(3):
#         env.reset(seed=episode)
#         for _ in range(5):
#             if env.agents:
#                 agent = env.agent_selection
#                 action = env.action_space(agent).sample()
#                 env.step(action)
#                 _ = env.observe(agent)
#                 _ = env.last(observe=True)
#     assert True
#
#
# # ====== T13: High agent count (100) stress test ======#
# def test_high_agent_count_stability():
#     from Failure_API.src.failure_api.wrappers.communication_wrapper import CommunicationWrapper
#     from Failure_API.src.failure_api.communication_models.probabilistic_model import ProbabilisticModel
#     try:
#         env = simple_spread_v3.env(N=100)
#         model = ProbabilisticModel(agent_ids=env.possible_agents, failure_prob=0.3)
#         env = CommunicationWrapper(env, failure_models=[model])
#         env.reset(seed=999)
#         agent = env.agent_selection
#         obs = env.observe(agent)
#         assert isinstance(obs, dict)
#     except Exception as e:
#         assert False, f"Crash on 100 agents: {str(e)}"
#
#
# # ====== T14: Observation corruption edge case ======#
# def test_noise_preserves_zeros():
#     import numpy as np
#     from Failure_API.src.failure_api.noise_models.gaussian_noise import GaussianNoiseModel
#     model = GaussianNoiseModel(std=0.1)
#     obs = np.array([0.0, 1.0, 0.0])
#     noisy = model.apply(obs.copy())
#     assert noisy[0] == 0.0 and noisy[2] == 0.0, f"Zeros were corrupted: {noisy}"
#
#
# # ====== T15: Model update mismatch with agent_ids ======#
# def test_model_agentid_mismatch_handling():
#     from Failure_API.src.failure_api.communication_models.distance_based_model import DistanceModel
#     import numpy as np
#     try:
#         model = DistanceModel(agent_ids=["fake_agent_1", "fake_agent_2"], distance_threshold=1.0,
#                               pos_fn=lambda a: np.zeros(2))
#         model.update_connectivity(None)  # Force crash by passing None instead of comms matrix
#         assert False, "Expected failure due to bad input"
#     except Exception:
#         assert True
#
#
# # ====== T16: Runtime and Memory Profiling (AEC) ======#
# def test_aec_runtime_memory_profiling():
#     import time
#     import psutil
#     from Failure_API.src.failure_api.wrappers.communication_wrapper import CommunicationWrapper
#     from Failure_API.src.failure_api.communication_models.probabilistic_model import ProbabilisticModel
#
#     env = simple_spread_v3.env(N=10, max_cycles=20)
#     model = ProbabilisticModel(agent_ids=env.possible_agents, failure_prob=0.2)
#     env = CommunicationWrapper(env, failure_models=[model])
#     env.reset(seed=42)
#
#     process = psutil.Process()
#     mem_before = process.memory_info().rss / 1e6
#
#     step_times = []
#     for _ in range(25):
#         if env.agents:
#             a = env.agent_selection
#             act = env.action_space(a).sample()
#             t0 = time.perf_counter()
#             env.step(act)
#             t1 = time.perf_counter()
#             step_times.append((t1 - t0) * 1000)
#
#     mem_after = process.memory_info().rss / 1e6
#     print(f"Average step time: {sum(step_times)/len(step_times):.3f} ms")
#     print(f"Memory overhead: {mem_after - mem_before:.3f} MB")
#
# # ====== T17: Parallel Multi-Wrapper Test (All Models) ======#
# def test_parallel_multi_model_combo():
#     from pettingzoo.utils.conversions import aec_to_parallel
#     from Failure_API.src.failure_api.wrappers.communication_wrapper import CommunicationWrapper
#     from Failure_API.src.failure_api.communication_models.delay_based_model import DelayBasedModel
#     from Failure_API.src.failure_api.communication_models.markov_model import BaseMarkovModel
#     from Failure_API.src.failure_api.communication_models.signal_based_model import SignalBasedModel
#     from Failure_API.src.failure_api.communication_models.distance_based_model import DistanceModel
#     import numpy as np
#     import time
#     import psutil
#
#     env = simple_spread_v3.env(N=15)
#     agent_ids = env.possible_agents
#
#     def dummy_pos_fn(agent):
#         return np.random.uniform(0, 1, size=2)
#
#     delay_model = DelayBasedModel(agent_ids=agent_ids, min_delay=1, max_delay=3)
#     markov_model = BaseMarkovModel(agent_ids=agent_ids)
#     signal_model = SignalBasedModel(agent_ids=agent_ids, pos_fn=lambda: {a: dummy_pos_fn(a) for a in agent_ids})
#     distance_model = DistanceModel(agent_ids=agent_ids, distance_threshold=1.0, pos_fn=dummy_pos_fn)
#
#     models = [delay_model, markov_model, signal_model, distance_model]
#
#     env = CommunicationWrapper(env, failure_models=models)
#     env = aec_to_parallel(env)
#     obs, _ = env.reset(seed=123)
#
#     process = psutil.Process()
#     mem_before = process.memory_info().rss / 1e6
#
#     step_times = []
#     for _ in range(10):
#         actions = {agent: env.action_space(agent).sample() for agent in env.agents}
#         t0 = time.perf_counter()
#         env.step(actions)
#         t1 = time.perf_counter()
#         step_times.append((t1 - t0) * 1000)
#
#     mem_after = process.memory_info().rss / 1e6
#
#     print(f"[Multi-Wrapper] Avg Step Time: {sum(step_times)/len(step_times):.2f} ms")
#     print(f"[Multi-Wrapper] Mem Usage: Δ{mem_after - mem_before:.2f} MB")
#
