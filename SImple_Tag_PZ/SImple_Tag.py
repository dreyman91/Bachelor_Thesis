from pettingzoo.mpe.simple_tag import env

import numpy as np

env = env(render_mode=None)
env.reset()

agent = env.agent_selection

initial_observation = env.observe(agent)

action = env.action_space(agent).sample()
env.step(action)

new_observation = env.observe(agent)

obsevation_analysis = {
    "Agent": agent,
    "Initial Observation Shape": np.shape(initial_observation),
    "Initial Observation Sample": initial_observation[:5],
    "New Observation Shape": np.shape(new_observation),
    "New Observation Sample": new_observation[:5],
}

obsevation_analysis