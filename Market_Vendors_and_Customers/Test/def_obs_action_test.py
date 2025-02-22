import unittest
import numpy as np
from Market_Vendors_and_Customers.market_aec import MarketAECEnv  # Import your environment class
from gymnasium import spaces


def test_environment():
    print("\n🔹 Initializing Environment...\n")

    # Create an instance of the environment
    env = MarketAECEnv(vendors=3, customers=3)

    print("\n✅ Environment Initialized Successfully!")

    # Test agent formatting function
    print("\n🔹 Testing Agent Name Formatting...\n")
    assert env.format_agent_name("customer", 0) == "customer_0"
    assert env.format_agent_name("vendor", 2) == "vendor_2"
    print("✅ Agent naming is consistent!")

    # Test reset function
    print("\n🔹 Testing Reset Function...\n")
    obs, _ = env.reset()
    assert isinstance(obs, dict), "Reset did not return a dictionary of observations!"
    assert len(obs) > 0, "Observations should not be empty!"
    print("✅ Reset function executed correctly!")

    # Test action space retrieval
    print("\n🔹 Testing Action Spaces...\n")
    for agent in env.agents:
        action_space = env.action_space(agent)
        assert isinstance(action_space, spaces.Discrete), f"Action space for {agent} should be Discrete!"
        print(f"✅ {agent} - Action Space: {action_space}")

    # Test observation space retrieval
    print("\n🔹 Testing Observation Spaces...\n")
    for agent in env.agents:
        obs_space = env.observation_space(agent)
        assert isinstance(obs_space, spaces.Box), f"Observation space for {agent} should be Box!"
        print(f"✅ {agent} - Observation Space: {obs_space}")

    # Test invalid agent lookup
    print("\n🔹 Testing Invalid Agent Handling...\n")
    try:
        env.action_space("invalid_agent_1")
    except KeyError as e:
        print(f"✅ Correctly handled invalid agent in action_space(): {e}")

    try:
        env.observation_space("invalid_agent_2")
    except KeyError as e:
        print(f"✅ Correctly handled invalid agent in observation_space(): {e}")

    print("\n🎉 All Tests Passed Successfully!")


if __name__ == "__main__":
    test_environment()
