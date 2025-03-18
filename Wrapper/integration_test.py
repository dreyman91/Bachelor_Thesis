from pettingzoo.test import api_test
from pettingzoo.mpe import simple_tag_v3
from Wrapper.comm_failure_api_v1 import (GaussianNoise,LaplacianNoise,CustomNoise,CommunicationModels,
                                         CommunicationFailure,ProbabilisticModel,DistanceModel,
                                         MatrixModel,NoiseModel,ActiveCommunication)
import numpy as np

env = simple_tag_v3.env(max_cycles=5)
env.reset()

print("Reward dict keys:", list(env._cumulative_rewards.keys()))
print("Terminations dict keys:", list(env.terminations.keys()))
print("Truncations dict keys:", list(env.truncations.keys()))


wrapped_env = CommunicationFailure(env)
print("Possible agents:", wrapped_env.possible_agents)
api_test(wrapped_env)
wrapped_env.close()
print("API Test succeeded")

