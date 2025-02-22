import unittest
import numpy as np
from Market_Vendors_and_Customers.market_aec import MarketAECEnv  # Import your environment class


def test_environment():
    """Runs a test cycle for the MarketAECEnv."""
    print("\n===== Starting Environment Test =====")

    # Initialize the environment
    env = MarketAECEnv(vendors=3, customers=3)
    obs, _ = env.reset()

    print("\n[TEST] Environment Reset Complete")
    print("[TEST] Agents:", env.agents)
    print("[TEST] Initial Observations:", obs)

    # Run a few steps
    for _ in range(5):  # Run 5 steps
        agent = env.agent_selection
        action = env.action_space(agent).sample()  # Sample a random action

        print(f"\n[TEST] Agent {agent} takes action {action}")
        obs, reward, terminations, truncations, infos = env.step(action)

        print(f"[TEST] Reward for {agent}: {reward}")
        print(f"[TEST] Terminations: {terminations}")
        print(f"[TEST] Truncations: {truncations}")
        print(f"[TEST] Infos: {infos}")

        if not env.agents:
            print("\n[TEST] All agents are done. Ending test.")
            break  # Stop if all agents are removed

    print("\n===== Test Completed Successfully =====")


if __name__ == "__main__":
    test_environment()
