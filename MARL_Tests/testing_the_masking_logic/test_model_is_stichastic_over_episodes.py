import numpy as np
from mpe2 import simple_spread_v3
from pettingzoo.utils.conversions import aec_to_parallel
from failure_api.wrappers import CommunicationWrapper
from failure_api.communication_models import ProbabilisticModel, DistanceModel, BaseMarkovModel


def evaluate_fixed_seed_rewards(model_name, trained_model, episodes=10):
    """Run environment multiple times with fixed seed and return total rewards."""
    rewards = []

    for ep in range(episodes):
        env = simple_spread_v3.env(N=3, max_cycles=25)
        env.reset(seed=42)  # fixed seed

        agent_ids = env.possible_agents

        if model_name == "probabilistic":
            failure_model = ProbabilisticModel(agent_ids=agent_ids, failure_prob=0.5)
        elif model_name == "distance":
            def get_positions(env):
                return {agent: env.env.state()[i][:2] for i, agent in enumerate(env.possible_agents)}

            failure_model = DistanceModel(agent_ids=agent_ids, distance_threshold=1.0,
                                          pos_fn=lambda: get_positions(env))
        elif model_name == "markov":
            P = np.array([[0.8, 0.2], [0.2, 0.8]])
            failure_model = BaseMarkovModel(agent_ids=agent_ids, transition_probabilities=P)
        else:
            raise ValueError("Unknown model")

        wrapped = CommunicationWrapper(env, failure_models=[failure_model])
        env = aec_to_parallel(wrapped)
        obs, _ = env.reset(seed=42)

        total_reward = 0
        for _ in range(25):
            actions = {
                agent: trained_model.predict(
                    obs[agent][agent],  # assumes wrapped obs
                    deterministic=True
                )[0]
                for agent in env.agents
            }
            obs, rews, terms, truncs, infos = env.step(actions)
            total_reward += sum(float(r) for r in rews.values())

            if all(terms.values()):
                break

        rewards.append(total_reward)
    return rewards


def test_model_is_stochastic_over_episodes():
    from stable_baselines3 import DQN
    model_path = r"C:\Users\koste\venv\Bachelor_Thesis\MARL_Tests\evaluations\.iql_training\simple_spread_v3_model_0.zip"
    trained_model = DQN.load(model_path)
    for model_name in ["probabilistic", "distance", "markov"]:
        rewards = evaluate_fixed_seed_rewards(model_name, trained_model, episodes=10)
        unique_rewards = len(set(rewards))
        print(f"{model_name}: unique rewards = {unique_rewards}")

        # Fail the test if all rewards are the same
        assert unique_rewards > 1, f"{model_name} model is deterministic! Rewards identical: {rewards}"

