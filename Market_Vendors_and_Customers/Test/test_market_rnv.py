import pytest
from pettingzoo.test import parallel_api_test
from Market_Vendors_and_Customers.market_aec import MarketAECEnv
import numpy as np

def test_environment_api():
    env = MarketAECEnv()
    parallel_api_test(env, num_cycles=100)

# Test reset functionality
def test_reset():
    env = MarketAECEnv()
    obs, infos = env.reset()

    # Check that observations are a dictionary
    assert isinstance(obs, dict)
    # Check that all possible agents have observations
    assert set(obs.keys()) == set(env.possible_agents)
    # Check observation space compliance
    for agent, ob in obs.items():
        assert env.observation_space(agent).contains(ob)

    #check that infos are a dictionary
    assert isinstance(infos, dict)
    assert set(infos.keys()) == set(env.possible_agents)
