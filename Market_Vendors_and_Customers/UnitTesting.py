from Market_Vendors_and_Customers.market_aec import MarketAECEnv  # Import your environment class
from pettingzoo.test import api_test
def test_pettingzoo_compliance():
    """Test if the environment complies with the PettingZoo API."""
    env = MarketAECEnv()
    api_test(env)
    print("✅ Environment is PettingZoo-compliant")

if __name__ == "__main__":
    test_pettingzoo_compliance()

