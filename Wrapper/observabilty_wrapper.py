"""
    A flexible wrapper for PettingZoo multi-agent environments that allows:
    
    1️⃣ **Dynamic Observation Masking**: Randomly hides parts of an agent's observation during each step.
    2️⃣ **Agent Visibility Control**: Controls the probability of hiding other agents' positions from each agent.
    3️⃣ **Landmark Visibility Control**: Controls the probability of hiding landmark positions from agents.
    4️⃣ **General Communication Failures**: Introduces random observation loss across all components.

    🔹 This wrapper is designed to introduce **partial observability** in multi-agent reinforcement learning (MARL).
    🔹 Unlike traditional wrappers that statically remove observations, this one applies **probabilistic masking** dynamically.
    🔹 Works **seamlessly with any PettingZoo environment**, even if the environment **does not have landmarks**.

    🛠️ **How It Works**
    - When an agent observes the environment, their observation consists of:
      - Their own position and velocity.
      - Other agents' positions (last 8 values in the vector).
      - Landmark positions (middle values in the vector).
    - This wrapper **dynamically masks** certain parts of the observation:
      - 🔴 **Agent Masking (`agent_hide_prob`)**: Controls the probability of hiding other agents' positions.
      - 🟠 **Landmark Masking (`landmark_hide_prob`)**: Controls the probability of hiding landmark positions.
      - 🟢 **General Failure (`dynamic_failure_prob`)**: Introduces random observation loss across all components.

    📌 **Arguments:**
    - `env` (PettingZoo environment): The multi-agent environment to wrap.
    - `agent_hide_prob` (float, 0.0 to 1.0): Probability of masking agent-related observations per step.
    - `landmark_hide_prob` (float, 0.0 to 1.0): Probability of masking landmark-related observations per step.
    - `dynamic_failure_prob` (float, 0.0 to 1.0): Probability of randomly removing any part of an agent’s observation.

    📌 **Usage Example:**
    ```python
    import pettingzoo.mpe.simple_spread_v3 as simple_spread
    env = simple_spread.env()
    wrapped_env = DynamicObservabilityWrapper(env, agent_hide_prob=0.3, landmark_hide_prob=0.5, dynamic_failure_prob=0.2)
    ```

    📌 **Why Use This?**
    - Introduces **partial observability** to test robustness in multi-agent systems.
    - Simulates **communication failures** where agents lose information randomly.
    - Works with **any multi-agent PettingZoo environment**, including those without landmarks.
    """
from pettingzoo.utils import BaseWrapper
import numpy as np


class DynamicObservabilityWrapper(BaseWrapper):

    def __init__(self, env, agent_hide_prob = 0.0, dynamic_failure_prob=0.0):
        super().__init__(env)
        self.agent_hide_prob = agent_hide_prob
        self.dynamic_failure_prob = dynamic_failure_prob

    def reset(self, seed=None, options= None):
        """
        Resets the environment and applies modified observations.

        This function calls the reset method of the base environment
        and returns a modified initial observation for all agents.
        """
        observations, _ = self.env.reset(seed=seed, options=options)
        return self.modify_observations(observations)
    def step(self, actions):
        """
        Executes a step in the environment while applying dynamic masking.
        """
        observations, rewards, terminations, truncations, infos = self.env.step(actions)
        return self.modify_observations(observations), rewards, terminations, truncations, infos

    def modify_observations(self, observations):
        """
        Dynamically modifies agent observations based on masking probabilities.

        The observation structure varies by environment, but typically includes:
        - Agent's own position and velocity.
        - Other agents' positions (last 8 values in observation vector).
        - Landmark positions (middle section of the observation vector).

        This function applies:
        - **Agent hiding (`agent_hide_prob`)**: Removes other agents' positions dynamically.
        - **Landmark hiding (`landmark_hide_prob`)**: Removes landmark positions dynamically.
        - **General failures (`dynamic_failure_prob`)**: Randomly masks any observation component.

        Returns:
        - A modified dictionary of observations with certain values hidden based on probabilities.
        """
        modified_obs = {}

        for agent, agent_obs in observations.items():
            obs_array = np.array(agent_obs)
            num_obs = len(obs_array)

            # Apply dynamic hiding

            if self.agent_hide_prob > 0 and num_obs >= 8:
                mask = np.random.rand(8) > self.agent_hide_prob
                obs_array[-8:] *= mask # Apply mask to the last 8 values

            if self.dynamic_failure_prob > 0:
                mask = np.random.rand(*obs_array.shape) > self.dynamic_failure_prob
                obs_array *= mask # Failure based masking

            modified_obs[agent] = obs_array.tolist()
        return modified_obs






