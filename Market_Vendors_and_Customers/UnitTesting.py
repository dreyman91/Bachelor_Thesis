
import unittest
from Market_Vendors_and_Customers.Market_AEC import Market_AEC_Env, Product
import numpy as np


class MultiAgentTest(unittest.TestCase):

    def setUp(self):
        self.env = Market_AEC_Env(vendors=2, customers=2, admin=1)
        self.customer = "customer_0"
        self.vendor = "vendor_0"
        self.admin = "admin_0"

        self.product = Product(product_id=1, price=50, name="Laptop", category="Electronics", stock=10)
        self.env.vendors_products[self.vendor] = [self.product]
        self.env.customer_budgets[self.customer] = 100

        #Test 1, Purchase
    def test_customer_vendor_transaction(self):
        self.env.purchase(self.customer, self.vendor, self.product)

        self.assertEqual(self.product.stock, 9,  "Stock should decrease after piurchase")
        self.assertEqual(self.env.customer_budgets[self.customer], 50, "Budge should decrease after purchase")
        self.assertEqual(self.env.vendor_revenue[self.vendor], 50, "vendor revenue should increase")

    def test_admin_penalizing_vendors(self):
        overpriced_product = Product(product_id=2, price=5000, name="Luxury Watch", category="Accessories", stock=5)
        self.env.vendors_products[self.vendor] = [overpriced_product]

        self.env.fine()

        self.assertGreater(self.env.violations.get(self.vendor, 0), 0, "Vendor should be penalized for overpricing")

 ###### ✅ TEST 3: Environment Step Function ######
    def test_environment_step(self):
        """Test that agents take turns correctly in step function"""
        initial_agent = self.env.agent_selection
        self.env.step(0)  # Take an action
        next_agent = self.env.agent_selection

        self.assertNotEqual(initial_agent, next_agent, "Agent should switch after step")

    ###### ✅ TEST 4: Reward System Validation ######
    def test_rewards(self):
        """Test if rewards are correctly assigned after transactions"""
        self.env.purchase(self.customer, self.vendor, self.product)

        self.assertGreater(self.env.rewards[self.customer], 0, "Customer should receive a positive reward for purchasing")
        self.assertGreater(self.env.rewards[self.vendor], 0, "Vendor should receive a positive reward for selling")

    ###### ✅ TEST 5: Edge Case - Buying when Out of Budget ######
    def test_purchase_out_of_budget(self):
        """Test if a customer cannot buy when out of budget"""
        expensive_product = Product(product_id=3, price=500, name="Expensive TV", category="Electronics", stock=5)
        self.env.vendors_products[self.vendor] = [expensive_product]
        self.env.customer_budgets[self.customer] = 100  # Not enough budget

        self.env.purchase(self.customer, self.vendor, expensive_product)

        self.assertEqual(self.env.customer_budgets[self.customer], 100, "Budget should not change")
        self.assertEqual(expensive_product.stock, 5, "Stock should remain the same")

    ###### ✅ TEST 6: Admin Monitoring Market Trends ######
    def test_monitor_market(self):
        """Test if the admin can monitor market trends"""
        market_info = self.env.monitor_market()
        self.assertIsNotNone(market_info, "Market monitoring should return valid data")

    ###### ✅ TEST 7: Vendor Adjusting Prices Based on Market Trends ######
    def test_vendor_adjust_prices(self):
        """Test if vendor adjusts prices based on market conditions"""
        self.env.adjust_prices(self.vendor)

        adjusted_price = self.env.vendors_products[self.vendor][0].price
        self.assertNotEqual(adjusted_price, 50, "Vendor price should be adjusted dynamically")

    ###### ✅ TEST 8: Market Dynamics - Vendors Running Out of Stock ######
    def test_vendor_out_of_stock(self):
        """Test if vendor becomes inactive when out of stock"""
        self.env.vendors_products[self.vendor] = []  # No stock left
        self.env.update_status(self.vendor)

        self.assertEqual(self.env.vendors_status[self.vendor], "inactive", "Vendor should become inactive when out of stock")

    ###### ✅ TEST 9: Admin Policy Enforcement ######
    def test_enforce_policies(self):
        """Test if admin properly enforces policies"""
        self.env.enforce_policies()

        self.assertIsInstance(self.env.vendor_penalties, dict, "Vendor penalties should be a dictionary")
        self.assertTrue(all(penalty >= 0 for penalty in self.env.vendor_penalties.values()), "Penalties should be non-negative")

    ###### ✅ TEST 10: Agent Step Transitions ######
    def test_step_transitions(self):
        """Test that environment steps correctly transition between agents"""
        for _ in range(5):
            prev_agent = self.env.agent_selection
            self.env.step(0)  # Take action
            new_agent = self.env.agent_selection
            self.assertNotEqual(prev_agent, new_agent, "Agents should take turns correctly")


if __name__ == "__main__":
    unittest.main(argv=[''], verbosity=2, exit=False)

