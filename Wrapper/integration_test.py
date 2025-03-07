from pettingzoo.mpe import simple_spread_v3
from Wrapper.comm_failure_api import CommunicationFailure

env = simple_spread_v3.parallel_env()
wrapped_env = CommunicationFailure(env, comms_model="probabilistic", failure_prob=0.3, seed=42)

observations, infos = wrapped_env.reset()

for _ in range(5):  # Short loop
    actions = {agent: wrapped_env.action_space(agent).sample() for agent in wrapped_env.agents}
    print(f"Actions BEFORE step: {actions}")  # KEY PRINT STATEMENT
    observations, rewards, terminations, truncations, infos = wrapped_env.step(actions)
    print(f"Actions AFTER step.: {actions}")  # KEY PRINT STATEMENT

    if all(terminations.values()) or all(truncations.values()):
        break
wrapped_env.close()