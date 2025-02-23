import pandas as pd
from tabulate import tabulate
import pettingzoo.mpe.simple_spread_v3 as simple_spread
from Wrapper.observabilty_wrapper import DynamicObservabilityWrapper
import numpy as np

raw_env = simple_spread.parallel_env()
masked_env = DynamicObservabilityWrapper(
    simple_spread.parallel_env(),
    agent_hide_prob=0.3,
    dynamic_failure_prob=0.2
)

def extract_observation(observation, num_agents, num_landmarks=3):
    """
    Dynamically extracts position, velocity, other agents' relative positions, and communication features.
    """
    obs = np.array(observation)

    if len(obs)== 0:
        return {
            "Position": [None, None],
            "Velocity": [None, None],
            "Other Agents": [None] * (2 * (num_agents - 1)),
            "Communication": [None] * (num_agents - 1)
        }

    idx = 0

    pos = obs[idx:idx + 2]
    idx += 2

    vel = obs[idx:idx + 2]
    idx += 2

    idx += 2 * num_landmarks

    other_agents = obs[idx:idx + (2 * (num_agents - 1))]
    idx += (2 * (num_agents - 1))

    communication = obs[idx:idx +  (num_agents - 1)]

    return {
        "Position": pos.tolist(),
        "Velocity": vel.tolist(),
        "Other_Agents": other_agents.tolist(),
        "Communication": communication.tolist()
    }

def collect_observations(env, filename, wrapped=False):
    storage = []
    observations = env.reset()
    num_agents = len(observations)

    if not hasattr(env, "possible_agents") or not env.possible_agents:
        print("No agents found.")
        return

    for step in range(50):
        actions = {agent: env.action_space(agent).sample() for agent in env.possible_agents}
        observations, rewards, terminations, truncations, infos = env.step(actions)

        for agent in env.possible_agents:
            obs = observations.get(agent, [])
            extracted = extract_observation(obs, num_agents)

            storage.append({
                "Step": step,
                "Agent": agent,
                "Position_X": round(extracted["Position"][0], 2) if extracted["Position"][0] is not None else None,
                "Position_Y": round(extracted["Position"][1], 2) if extracted["Position"][1] is not None else None,
                "Velocity_X": round(extracted["Velocity"][0], 2) if extracted["Velocity"][0] is not None else None,
                "Velocity_Y": round(extracted["Velocity"][1], 2) if extracted["Velocity"][1] is not None else None,
                "Other_Agent_1_X": round(extracted["Other_Agents"][0], 2) if len(
                    extracted["Other_Agents"]) > 0 else None,
                "Other_Agent_1_Y": round(extracted["Other_Agents"][1], 2) if len(
                    extracted["Other_Agents"]) > 1 else None,
                "Other_Agent_2_X": round(extracted["Other_Agents"][2], 2) if len(
                    extracted["Other_Agents"]) > 2 else None,
                "Other_Agent_2_Y": round(extracted["Other_Agents"][3], 2) if len(
                    extracted["Other_Agents"]) > 3 else None,
                "Comm_1": round(extracted["Communication"][0], 2) if len(extracted["Communication"]) > 0 else None,
                "Comm_2": round(extracted["Communication"][1], 2) if len(extracted["Communication"]) > 1 else None,
                "Rewards": rewards.get(agent, 0),
                "Masked": wrapped
            })

        if all(terminations.values()) or all(truncations.values()):
            break

    df = pd.DataFrame(storage)
    df.to_csv(filename, index=False)

collect_observations(raw_env, "raw_observations.csv", wrapped=False)
collect_observations(masked_env, "masked_observations.csv", wrapped=True)