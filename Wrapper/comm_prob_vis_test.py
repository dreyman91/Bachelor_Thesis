import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pettingzoo.mpe import simple_spread_v3
from Wrapper.comm_failure_api import CommunicationFailure
from Wrapper.parallel_prob_test import wrapped_env

#-----Simulation Parameters------------#
failure_prob = 0.3
num_trials = 100
max_cycles = 25
seed = 42

#--- Create Environment ---#
env = simple_spread_v3.parallel_env()
wrapped_env = CommunicationFailure(
    env,
    comms_model="probabilistic",
    failure_prob=failure_prob,
    seed=seed,
)

#--- Data Structure ---#
num_agents = len(wrapped_env.possible_agents)
comm_attempts = np.zeros((num_agents, num_agents))
comm_failures = np.zeros((num_agents, num_agents))
agent_mapping = {agent: i for i, agent in enumerate(wrapped_env.possible_agents)}

#---- Run Simulation ----#
for _ in range(num_trials):
    observations, infos = wrapped_env.reset()
    terminations = {agent: False for agent in wrapped_env.possible_agents}
    truncations = {agent: False for agent in wrapped_env.possible_agents}

    for _ in range(max_cycles):
        actions = {agent: wrapped_env.action_space(agent).sample() for agent in wrapped_env.agents}
        observation, rewards, terminations, truncations, infos = wrapped_env.step(actions)

        for agent in wrapped_env.agents:
            agent_index = agent_mapping[agent]

            # Check available agents before comms simulation
            for other_agent in wrapped_env.agents:
                other_agent_index = agent_mapping[other_agent]
                if agent != other_agent: # Prohibit self comm
                    comm_attempts[agent_index, other_agent_index] += 1

                    # Check if communication failed between agents
                    if not wrapped_env.active_comms[other_agent][agent]:
                        comm_failures[agent_index, other_agent_index] += 1

        observations = observations # Update Obs
        if all(terminations.values()) or all(truncations.values()):
            break

    wrapped_env.close()

#---- Calculate Empirical Failure Probabilities ---#
fail_rates = np.divide(comm_failures, comm_attempts, out=np.zeros_like(comm_failures), where=comm_attempts!=0)

data = []
for i in range(num_agents):
    for j in range(num_agents):
        data.append({
            'Sending Agent': wrapped_env.possible_agents[i],
            'Received Agent': wrapped_env.possible_agents[j],
            'Failure Rate': fail_rates[i, j],
            'Attempts': comm_attempts[i, j],
            'Failures.': comm_failures[i, j]
        })

df = pd.DataFrame(data)
print(df)
