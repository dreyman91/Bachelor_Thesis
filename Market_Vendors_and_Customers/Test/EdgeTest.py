from Market_Vendors_and_Customers.market_aec import MarketAECEnv  # Import your environment class
import pytest
import numpy as np
from gymnasium import spaces

def test_market_env_initialization():
    """Test initialization of MarketAECEnv"""

    try:
        env = MarketAECEnv(vendors=3, customers=3)
        print("[PASS] Environment initialized successfully.")

        # 1. Test Agents Initialization
        assert len(env.agents) == 6, "[FAIL] Agents list is incorrect."
        assert len(env.possible_agents) == 6, "[FAIL] Possible agents list is incorrect."
        print(f"[PASS] Agents Initialized Correctly: {env.agents}")

        # 2. Test Market Attributes
        assert len(env.vendors_products) == 3, "[FAIL] Vendors products not initialized correctly."
        assert len(env.market_prices) == 3, "[FAIL] Market prices not initialized correctly."
        assert len(env.market_price_history) == 3, "[FAIL] Market price history not initialized correctly."
        print("[PASS] Market attributes initialized correctly.")

        # 3. Test Reward System
        assert len(env.rewards) == 6, "[FAIL] Rewards dictionary is incorrect."
        assert len(env._cumulative_rewards) == 6, "[FAIL] Cumulative rewards dictionary is incorrect."
        print("[PASS] Rewards initialized correctly.")

        # 4. Test Observation and Action Spaces
        assert isinstance(env.observation_spaces, dict), "[FAIL] Observation spaces are not a dictionary."
        assert isinstance(env.action_spaces, dict), "[FAIL] Action spaces are not a dictionary."
        print("[PASS] Spaces initialized correctly.")

        print("==== MarketAECEnv Initialization Test PASSED ====")

    except Exception as e:
        print(f"[FAIL] Initialization failed: {e}")

# Run the test
if __name__ == "__main__":
    test_market_env_initialization()
