# Loading the built-in environment from pettingzoo
import pandas as pd
import numpy as np
from pettingzoo.mpe import simple_adversary_v3

# load the environment
env = simple_adversary_v3.env()

# Reset environment
env.reset()

# Step through the environment using agent iter()

for agent in env.agent_iter():
    print("Current Agent: ", agent)
    obs, reward, done, truncated, info = env.last()  # get last agents info

    if done or truncated:
        action = None
    else:
        action = env.action_space(agent).sample()  # Select an action
    env.step(action)

# Agent moving to next state
for agent in env.agent_iter():

    obs, reward, done, truncated, info = env.last()
    if done or truncated:
        action = None
    else:
        action = env.action_space(agent).sample()
    env.step(action)

env.close()

# env.reset()
# obs, reward, done, truncated, info = env.last()
# print("Last agent's observation: ", obs)
# print("Last agent's reward: ", reward)
# print("Last agent is done?: ", done)


# print(f"{agent} took action {action} -> Reward: {reward}")
# print("Active Agents Before Reset", env.agents)
# print("Possible  agents", env.possible_agents)
#
#
# print("Active Agents", env.agents)
# print("Possible  agents", env.possible_agents)
# print("Agent Termination status:", env.terminations)
# #
# obs = env.observe("agent_0")
# df_matrix = pd.DataFrame(obs)
# print("Observation of Agent 0", df_matrix)




