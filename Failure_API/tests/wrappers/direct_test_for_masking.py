import numpy as np
from mpe2 import simple_spread_v3
from pettingzoo.utils import aec_to_parallel
from Failure_API.src.failure_api.wrappers import CommunicationWrapper
from Failure_API.src.failure_api.communication_models import ProbabilisticModel


# def test_masking_logic(N=3, p=0.5, steps=5, seed=42):
#     """Stress test masking logic by asserting strict compliance"""
#     env = simple_spread_v3.env(N=N, max_cycles=steps)
#     agent_ids = env.possible_agents
#     comm_model = ProbabilisticModel(agent_ids=agent_ids, failure_prob=p)
#     wrapped_env = CommunicationWrapper(env, failure_models=[comm_model])
#     par_env = aec_to_parallel(wrapped_env)
#     obs, _ = par_env.reset(seed=seed)
#
#     for step in range(steps):
#         actions = {
#             agent: par_env.action_space(agent).sample()
#             for agent in par_env.agents
#         }
#         obs_dict, _, terminations, truncations, _ = par_env.step(actions)
#         obs_dict = {agent: wrapped_env.observe(agent) for agent in wrapped_env.agents}
#         comm_matrix = wrapped_env.comms_matrix.get_boolean_matrix().copy()
#
#         for receiver in par_env.agents:
#             obs = obs_dict[receiver]  # ✅ now correct
#             assert obs is not None, f"Observation is None for agent {receiver}"
#
#             if isinstance(obs, dict):
#                 for sender, vec in obs.items():
#                     if sender not in agent_ids:
#                         continue
#                     sender_idx = wrapped_env.agent_ids.index(sender)
#                     receiver_idx = wrapped_env.agent_ids.index(receiver)
#                     allowed = comm_matrix[sender_idx, receiver_idx]
#
#                     if sender != receiver:
#                         if allowed == 0:
#                             assert np.linalg.norm(vec) < 1e-6, (
#                                 f"[FAIL] Step {step}: {receiver} should not see {sender}, ‖obs‖={np.linalg.norm(vec):.4f}"
#                             )
#                         else:
#                             assert np.linalg.norm(vec) > 1e-6, (
#                                 f"[FAIL] Step {step}: {receiver} should see {sender}, but ‖obs‖={np.linalg.norm(vec):.4f}"
#                             )
#
#         if all(terminations.values()) or all(truncations.values()):
#             break
#
#     assert True  # ✅ All checks passed
#
#
# # Example: Run across scenarios
# def test_all_masking_conditions():
#     for N in [3, 10, 25]:
#         for p in [0.0, 0.25, 0.5, 0.75, 1.0]:
#             test_masking_logic(N=N, p=p, steps=10)

env = simple_spread_v3.env(N=3)
model = ProbabilisticModel(agent_ids=env.possible_agents, failure_prob=0.3)

wrapped = CommunicationWrapper(env, failure_models=[model])
obs, _ = wrapped.reset(seed=42)
def test_rng_determinism():
    env = simple_spread_v3.env(N=3)
    model = ProbabilisticModel(agent_ids=env.possible_agents, failure_prob=0.3)
    wrapped = CommunicationWrapper(env, failure_models=[model])

    obs1, _ = wrapped.reset(seed=42)
    r1 = wrapped.get_communication_state()

    obs2, _ = wrapped.reset(seed=42)
    r2 = wrapped.get_communication_state()

    assert np.allclose(r1, r2), "Connectivity matrix not reproducible!"


def _print_matrix(matrix, label):
    print(f"\n{label}:")
    print(np.array(matrix).astype(int))

def test_rng_reproducibility_failure_model():
    env = simple_spread_v3.env(N=3)
    model = ProbabilisticModel(agent_ids=env.possible_agents, failure_prob=0.5)
    wrapped = CommunicationWrapper(env, failure_models=[model])

    print("\n[RUN 1]")
    obs1, _ = wrapped.reset(seed=123)
    r1 = wrapped.get_communication_state()
    sample1 = model.rng.random()
    _print_matrix(r1, "Matrix 1")
    print(f"Sample RNG value 1: {sample1:.6f}")

    print("\n[RUN 2]")
    obs2, _ = wrapped.reset(seed=123)
    r2 = wrapped.get_communication_state()
    sample2 = model.rng.random()
    _print_matrix(r2, "Matrix 2")
    print(f"Sample RNG value 2: {sample2:.6f}")

    # Assert that matrices are exactly the same
    assert np.allclose(r1, r2), f"❌ Connectivity matrices not equal!\nMatrix1:\n{r1}\nMatrix2:\n{r2}"

    # Assert that RNG samples also match (internal state check)
    assert np.isclose(sample1, sample2), f"❌ RNG mismatch: {sample1:.6f} != {sample2:.6f}"

    print("\n✅ RNG is deterministic and model output is reproducible.")