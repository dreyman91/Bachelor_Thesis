import time
import psutil
import numpy as np
from mpe2 import simple_spread_v3
from pettingzoo.utils import aec_to_parallel
from failure_api.wrappers import CommunicationWrapper
from failure_api.communication_models import ProbabilisticModel


def benchmark_env(N, use_wrapper=True, max_cycles=25, seed=42):
    # Step 1: Create AEC base environment
    base_env = simple_spread_v3.env(N=N, max_cycles=max_cycles)

    if use_wrapper:
        model = ProbabilisticModel(agent_ids=base_env.possible_agents, failure_prob=0.3)
        base_env = CommunicationWrapper(base_env, failure_models=[model])

    # Step 2: Convert to parallel mode
    env = aec_to_parallel(base_env)
    env.reset(seed=seed)

    # Step 3: Measure memory before execution
    process = psutil.Process()
    mem_before = process.memory_info().rss / 1e6  # in MB

    # Step 4: Run a fixed number of steps and measure time
    step_times = []
    for _ in range(max_cycles):
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        start = time.perf_counter()
        env.step(actions)
        step_times.append((time.perf_counter() - start) * 1000)  # in ms

    mem_after = process.memory_info().rss / 1e6
    avg_step_time = np.mean(step_times)
    delta_mem = mem_after - mem_before

    return avg_step_time, delta_mem


agent_counts = [3, 10, 25, 50]

for N in agent_counts:
    base_time, base_mem = benchmark_env(N, use_wrapper=False)
    wrapped_time, wrapped_mem = benchmark_env(N, use_wrapper=True)

    overhead = ((wrapped_time - base_time) / base_time) * 100
    print(f"Agents: {N} | Step Time ↑ {overhead:.2f}% | Δ Memory: {wrapped_mem - base_mem:.3f} MB")


# import time
# import psutil
# import numpy as np
# from mpe2 import simple_spread_v3
# from failure_api.wrappers import CommunicationWrapper
# from failure_api.communication_models import ProbabilisticModel
#
#
# def benchmark_aec_env(N, use_wrapper=True, max_cycles=25, seed=42):
#     from mpe2 import simple_spread_v3
#     from failure_api.wrappers import CommunicationWrapper
#     from failure_api.communication_models import ProbabilisticModel
#     import time, psutil, numpy as np
#
#     env = simple_spread_v3.env(N=N, max_cycles=max_cycles)
#
#     if use_wrapper:
#         model = ProbabilisticModel(agent_ids=env.possible_agents, failure_prob=0.3)
#         env = CommunicationWrapper(env, failure_models=[model])
#
#     env.reset(seed=seed)
#
#     process = psutil.Process()
#     mem_before = process.memory_info().rss / 1e6  # in MB
#
#     step_times = []
#     for _ in range(max_cycles):
#         while env.agents:
#             agent = env.agent_selection
#
#             # ✅ PettingZoo-safe check for terminated agent
#             if env.terminations.get(agent, False) or env.truncations.get(agent, False):
#                 env.step(None)
#                 continue
#
#             action = env.action_space(agent).sample()
#             start = time.perf_counter()
#             env.step(action)
#             step_times.append((time.perf_counter() - start) * 1000)
#
#     mem_after = process.memory_info().rss / 1e6
#     avg_step_time = np.mean(step_times)
#     delta_mem = mem_after - mem_before
#
#     return avg_step_time, delta_mem
#
#
# agent_counts = [3, 10, 25, 50, 200]
#
# for N in agent_counts:
#     base_time, base_mem = benchmark_aec_env(N, use_wrapper=False)
#     wrapped_time, wrapped_mem = benchmark_aec_env(N, use_wrapper=True)
#
#     overhead = ((wrapped_time - base_time) / base_time) * 100
#     print(f"Agents: {N} | Step Time ↑ {overhead:.2f}% | Δ Memory: {wrapped_mem - base_mem:.3f} MB")

