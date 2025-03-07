import numpy as np
from Wrapper.large_scale_dummy import LargeScaleDummyEnv
from Wrapper.comm_failure_api import CommunicationFailure
import time as time



def measure_execution_time(num_agents=10):
    env = LargeScaleDummyEnv(num_agents)
    comms_matrix = np.random.uniform(0, 1, (num_agents, num_agents))
    np.fill_diagonal(comms_matrix, 1)  # Self-communication always enabled
    cf = CommunicationFailure(env, comms_matrix=comms_matrix, comms_model="matrix")

    actions = {agent: 1 for agent in env.possible_agents}  # Dummy actions

    # Measure step execution time
    start = time.time()
    cf.step(actions)
    end = time.time()
    print(f"Step execution time for {num_agents} agents: {end - start:.6f} seconds")

    # Measure communication update execution time
    start = time.time()
    cf._update_communication_state()
    end = time.time()
    print(f"Communication update execution time for {num_agents} agents: {end - start:.6f} seconds")


if __name__ == "__main__":
    for agents in [10, 50, 100, 500]:
        measure_execution_time(num_agents=agents)
