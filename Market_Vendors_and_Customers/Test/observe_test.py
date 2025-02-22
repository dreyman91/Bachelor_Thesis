import unittest
import numpy as np
from Market_Vendors_and_Customers.market_aec import MarketAECEnv  # Import your environment class

def test_observe():
    """Test the observe() function for edge cases and validation."""

    # Initialize environment
    env = MarketAECEnv(vendors=3, customers=3)
    print("\n✅ Environment Initialized")

    # Test 1: Observing before reset
    print("\n🔹 Test 1: Observing before reset()")
    agent = "customer_0"
    try:
        obs = env.observe(agent)
        print(f"✅ Observation before reset: {obs.shape} (Expected: {env.max_obs_size})")
    except Exception as e:
        print(f"❌ Error when observing before reset: {e}")

    # Reset environment
    env.reset()
    print("\n✅ Environment Reset")

    # Select a valid agent for testing
    test_agent = env.agents[0]

    # Test 2: Observing a valid agent
    print("\n🔹 Test 2: Observing a valid agent")
    try:
        obs = env.observe(test_agent)
        print(f"✅ Valid observation shape: {obs.shape} (Expected: {env.max_obs_size})")
    except Exception as e:
        print(f"❌ Error when observing valid agent: {e}")

    # Test 3: Removing an agent and trying to observe
    print("\n🔹 Test 3: Removing an agent mid-episode and observing")
    try:
        removed_agent = env.agents.pop(0)  # Remove first agent
        print(f"🛑 Removed agent: {removed_agent}")
        obs = env.observe(removed_agent)
        print(f"✅ Observation after removal: {obs.shape} (Expected: {env.max_obs_size}, should be blank)")
    except Exception as e:
        print(f"❌ Error when observing removed agent: {e}")

    # Test 4: Observing after step() removes an agent
    print("\n🔹 Test 4: Observing after step() removes an agent")
    try:
        # Simulate a step where an agent terminates
        for _ in range(len(env.agents)):
            agent = env.agent_selection
            action = env.action_space(agent).sample()
            env.step(action)
            if env.terminations.get(agent, False):  # If agent is terminated
                print(f"🛑 Step removed agent: {agent}")
                obs = env.observe(agent)
                print(f"✅ Observation after termination: {obs.shape} (Expected: {env.max_obs_size}, should be blank)")
    except Exception as e:
        print(f"❌ Error when observing after step removes agent: {e}")

    # Test 5: Checking observation shape consistency
    print("\n🔹 Test 5: Checking observation shape consistency")
    try:
        for agent in env.agents:
            obs = env.observe(agent)
            assert obs.shape == (env.max_obs_size,), f"❌ Mismatch in shape for {agent}: {obs.shape}"
        print("✅ All agent observations match expected shape.")
    except Exception as e:
        print(f"❌ Error in observation shape consistency: {e}")

    print("\n🎉 All tests completed!")

# Run the test
test_observe()
