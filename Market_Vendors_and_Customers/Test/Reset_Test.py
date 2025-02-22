from Market_Vendors_and_Customers.market_aec import MarketAECEnv  # Import your environment class


import pytest

def test_market_env_reset():
    """Test the reset function of MarketAECEnv"""

    try:
        env = MarketAECEnv(vendors=3, customers=3)
        env.reset()
        print("[PASS] Reset function executed successfully.")

        # 1. Ensure all agents are restored
        assert env.agents == env.possible_agents, "[FAIL] Agents were not restored correctly after reset."
        print(f"[PASS] Agents restored correctly: {env.agents}")

        # 2. Ensure rewards, terminations, and truncations are reset
        assert all(env.rewards[agent] == 0 for agent in env.agents), "[FAIL] Rewards were not reset correctly."
        assert all(not env.terminations[agent] for agent in env.agents), "[FAIL] Terminations were not reset correctly."
        assert all(not env.truncations[agent] for agent in env.agents), "[FAIL] Truncations were not reset correctly."
        print("[PASS] Rewards, terminations, and truncations reset correctly.")

        # 3. Call reset again to ensure stability
        env.reset()
        assert env.agents == env.possible_agents, "[FAIL] Reset failed on second call."
        print("[PASS] Reset function is stable when called multiple times.")

        print("==== MarketAECEnv Reset Test PASSED ====")

    except Exception as e:
        print(f"[FAIL] Reset function encountered an error: {e}")

# Run the test
if __name__ == "__main__":
    test_market_env_reset()
