import numpy as np
from mpe2 import simple_spread_v3
from Failure_API.src.utils.position_utils import make_position_fn
from Failure_API.src.failure_api.communication_models import DistanceModel, SignalBasedModel
from Failure_API.src.failure_api.wrappers.communication_wrapper import CommunicationWrapper
from pettingzoo.utils import aec_to_parallel
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import psutil
import os


def run_stress_test(model_type="distance", N=25, max_cycles=50):
    env = simple_spread_v3.env(N=N, max_cycles=max_cycles, continuous_actions=False)
    pos_fn = make_position_fn(env, slice_idx=(0, 2), debug=False)

    if model_type == "signal":
        pos_fn = make_position_fn(env, slice_idx=(0, 2), return_batch=True)  # For model
        pos_fn_single = make_position_fn(env, slice_idx=(0, 2))  # For step loop
        model = SignalBasedModel(
            agent_ids=env.possible_agents,
            pos_fn=pos_fn,
            min_strength=0.2
        )
    elif model_type == "distance":
        pos_fn = make_position_fn(env, slice_idx=(0, 2))
        pos_fn_single = pos_fn
        model = DistanceModel(
            agent_ids=env.possible_agents,
            distance_threshold=2.5,
            pos_fn=pos_fn,
            failure_prob=0.0,

        )
    else:
        raise ValueError("Model type must be either 'signal' or 'distance'")

    wrapped = CommunicationWrapper(env, failure_models=[model])
    parallel_env = aec_to_parallel(wrapped)
    obs, infos = parallel_env.reset(seed=42)

    step_times = []
    observe_times = []

    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / 1024 ** 2  # MB

    for _ in tqdm(range(max_cycles), desc=f"Stress Test with {model_type} Model (N={N})"):
        actions = {agent: parallel_env.action_space(agent).sample() for agent in parallel_env.agents}

        t0 = time.perf_counter()
        obs, rewards, terminations, truncations, infos = parallel_env.step(actions)
        t1 = time.perf_counter()
        step_times.append((t1 - t0) * 1000)  # in ms

        t_obs0 = time.perf_counter()
        for agent in parallel_env.agents:
            pos = pos_fn_single(agent)
            assert isinstance(pos, np.ndarray)
            assert pos.shape == (2,)

        t_obs1 = time.perf_counter()
        observe_times.append((t_obs1 - t_obs0) * 1000)

    mem_after = process.memory_info().rss / 1024 ** 2  # MB
    mem_delta = mem_after - mem_before

    return {
        "model": model_type,
        "agents": N,
        "avg_step_time_ms": np.mean(step_times),
        "avg_observe_time_ms": np.mean(observe_times),
        "memory_overhead_mb": mem_delta,
        "step_times": step_times,
        "observe_times": observe_times,
    }


results = []
for model_type in ["distance", "signal"]:
    for agent_count in [3, 10, 25, 50]:
        result = run_stress_test(model_type=model_type, N=agent_count, max_cycles=30)
        results.append(result)

df = pd.DataFrame(results)

