import pandas as pd
from tabulate import tabulate
from pettingzoo.mpe import simple_spread_v3
from Wrapper.observabilty_wrapper1 import DynamicObservabilityWrapper

env = simple_spread_v3.parallel_env(render_mode="human")

wrapped_env = DynamicObservabilityWrapper(env,
                                          agent_hide_prob=0.3, # 20% chance other agents are hidden
                                          dynamic_failure_prob=0.2 # 20% chance of losing an observation per step
                                          )

observations = wrapped_env.reset()

for step in range(10):
    print(f"\n Step {step + 1}")

    actions = {agent: wrapped_env.action_space(agent).sample() for agent in wrapped_env.agents}

    observations, reward, termination, truncation, info = wrapped_env.step(actions)

    if all(termination.values()) or all(truncation.values()):
        break

    for agent in wrapped_env.agents:
        print(f"🔹 Agent: {agent} | Reward: {reward} | Modified Obs: {observations[agent][:10]}...")

wrapped_env.close()