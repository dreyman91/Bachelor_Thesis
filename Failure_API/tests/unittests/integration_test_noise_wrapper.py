"""
To test whether Noise are applied to the observations when expected, and values are within bounds.
"""
import numpy as np
from numpy import dtype
from pettingzoo.utils import AECEnv
from gymnasium import spaces
from Failure_API.src.wrapper_api.models.noise_model import CustomNoise, GaussianNoise, LaplacianNoise
import unittest

from Failure_API.src.wrapper_api.failure_apis import NoiseWrapper


class DummyEnv(AECEnv):
    def __init__(self):
        self.possible_agents = ["a0", "a1", "a2"]
        self.agent_selection = "a0"
    def observe(self, agent):
        obs_dict = {
            "a0": np.array([0.6, 0.5], dtype=np.float32),
            "a1": np.array([0.7, 0.8], dtype=np.float32),
            "a2": np.array([0.9, 0.3], dtype=np.float32)
        }
        return obs_dict[agent]

    def observation_space(self, agent):
        return spaces.Dict({
            "a0": spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32),
            "a1": spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32),
            "a2": spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32),
        })

    def action_space(self, agent):
        return spaces.Discrete(2)

    def step(self, action):
        pass
    def reset(self):
        pass


env = DummyEnv()
def cm_noise(obs, space):
    ns = np.random.uniform(-0.1, 0.2, size=obs.shape).astype(np.float32)
    return obs + ns

model = CustomNoise(noise_fn=cm_noise)
wrpd_env = NoiseWrapper(env, noise_model = model)
nsy_obs = wrpd_env.observe("a0")

print(f"\n === Custom Noise === \n Obs: {nsy_obs},\n Shape: {nsy_obs.shape},\n Datatype: {nsy_obs.dtype}")

def gauss_ns(obs, space):
    return obs
model = GaussianNoise(mean=0.1, std=0.2)
wrpd_gs = NoiseWrapper(env, noise_model = model)
nsy_gs_obs = wrpd_gs.observe("a1")

print(f"=== Gaussian Noise === \n Obs: {nsy_gs_obs}")
