import time
import numpy as np
import psutil
import os
import pandas as pd

from mpe2 import simple_spread_v3
from pettingzoo.utils import aec_to_parallel
from Failure_API.src.failure_api.communication_models import ProbabilisticModel
from Failure_API.src.failure_api.wrappers import CommunicationWrapper


def run_perf_test_aec(agent_count):
    env = simple_spread_v3.env(N=agent_count, max_cycles=25)
    agent_ids = env.possible_agents
    model = ProbabilisticModel(agent_ids=agent_ids, failure_prob=0.3)
    env = CommunicationWrapper(env, failure_models=[model])
    env.reset(seed=42)

    step_times = []
    obs_times = []

    process = psutil.Process()
    mem_before = process.memory_info().rss / 1e6

    for _ in range(25 * agent_count):  # 25 full env cycles
        agent = env.agent_selection

        obs_start = time.perf_counter()
        _ = env.observe(agent)
        obs_end = time.perf_counter()

        action = env.action_space(agent).sample()

        step_start = time.perf_counter()
        env.step(action)
        step_end = time.perf_counter()

        obs_times.append((obs_end - obs_start) * 1000)
        step_times.append((step_end - step_start) * 1000)

    mem_after = process.memory_info().rss / 1e6

    avg_obs_time = np.mean(obs_times)
    avg_step_time = np.mean(step_times)
    mem_used = mem_after - mem_before

    print(f"\n--- AEC Mode Performance (N={agent_count}) ---")
    print(f"Avg step(): {avg_step_time:.3f} ms")
    print(f"Avg observe(): {avg_obs_time:.3f} ms")
    print(f"Memory overhead: {mem_used:.3f} MB")

    # Strict performance checks
    if agent_count == 5:
        assert avg_step_time < 1.5, "🚨 Step time too high for N=5"
        assert avg_obs_time < 2.0, "🚨 Observe too slow for N=5"
    elif agent_count == 25:
        assert avg_step_time < 3.5, "🚨 Step time too high for N=25"
        assert avg_obs_time < 4.0, "🚨 Observe too slow for N=25"
    elif agent_count == 50:
        assert avg_step_time < 6.0, "🚨 Step time too high for N=50"
        assert avg_obs_time < 7.5, "🚨 Observe too slow for N=50"
    elif agent_count == 100:
        assert avg_step_time < 9.0, "🚨 Step time too high for N=100"
        assert avg_obs_time < 10.5, "🚨 Observe too slow for N=100"

    assert mem_used < 3.5, f"🚨 Memory usage excessive: {mem_used:.2f} MB"


# Pytest will pick these up
def test_commwrapper_n5(): run_perf_test_aec(5)
def test_commwrapper_n25(): run_perf_test_aec(25)
def test_commwrapper_n50(): run_perf_test_aec(50)