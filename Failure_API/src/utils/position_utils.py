import numpy as np


def make_position_fn(env, key="position", slice_idx=(0, 2), debug=False, return_batch=False):
    """
    Creates a pos_fn(agent) or pos_fn() depending on return_batch.

    - If return_batch is False: returns a function that takes one agent and returns its position.
    - If return_batch is True: returns a function that returns a dict {agent_id: position}.
    """
    def single_pos_fn(agent):
        obs = env.observe(agent)
        if debug:
            print(f"[DEBUG] Observation for {agent}: {obs}")

        if isinstance(obs, dict):
            if key not in obs:
                raise ValueError(f"Key '{key}' not found in observation for agent {agent}")
            return np.asarray(obs[key])

        elif isinstance(obs, np.ndarray):
            if len(obs) < slice_idx[1]:
                raise ValueError(f"Observation for agent {agent} is too short: {obs}")
            return np.asarray(obs[slice_idx[0]:slice_idx[1]])

        else:
            raise TypeError(f"Unsupported observation type for agent {agent}: {type(obs)}")

    if not return_batch:
        return single_pos_fn

    # Return a batch-compatible version
    def batch_pos_fn():
        return {agent: single_pos_fn(agent) for agent in env.possible_agents}

    return batch_pos_fn


"""
==============USAGE ==============:
from position_utils import make_position_fn

pos_fn = make_position_fn(env)
model = DistanceModel(agent_ids=env.possible_agents, distance_threshold=5.0, pos_fn=pos_fn)


"""