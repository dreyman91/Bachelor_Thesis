import unittest
import numpy as np
from Market_Vendors_and_Customers.market_aec import Market_AEC_Env, Product

class Market_Vendors_and_Customers_and_UnitTesting(unittest.TestCase):
    def setUp(self):
        self.env = Market_AEC_Env(vendors=3, customers=3, admin=1)
        self.env.reset()
        self.agents = self.env.agents

        ## test Environment
        self.assertEqual(len(self.env.agents), 7)
        self.assertEqual(len(self.env.vendors_products), 3)
        self.assertIsInstance(self.env.market_prices, np.ndarray)

    def test_reset_fucntion(self):
        self.env.reset()
        self.assertEqual(self.env.agent_selection, self.env.agents[0])
        self.assertEqual(len(self.env.vendors_products), 3)

    def test_observation_fucntion(self):
        for agent in self.agents:
            obs = self.env.observe(agent)
            self.assertIsNotNone(obs)
            self.assertTrue(isinstance(obs, dict))

    def customer_purchase(self):
        customer = "customer_0"
        vendor = "vendor_0"
        product = self.env.vendors_products[vendor][0]
        initial_budget = self.env.customer_budgets[customer]
        initial_vendor_stock = product.stock
        self.env.purchase(customer, vendor, product)

        self.assertLess(self.env.customer_budgets[customer], initial_budget)
        self.assertLess(product.stock, initial_vendor_stock)

    def test_customer_check_market_trend(self):
        customer = "customer_0"
        self.env.check_market_trend(customer)

    def test_vendor_add_product(self):
        """Test if vendors can add new products."""
        vendor = "vendor_0"
        new_product = Product(4, 100, "Laptop", "Electronics", 5)

        if vendor not in self.env.vendors_products:
            print(f"Vendor {vendor} does not exist in the environment!")

        initial_count = len(self.env.vendors_products[vendor])
        self.env.add_product(vendor, new_product)  # <== Possible source of error

        final_count = len(self.env.vendors_products[vendor])
        self.assertEqual(final_count, initial_count + 1)  # Ensure product was added

    def test_vendor_update_stock(self):
        """Test if vendors can update stock correctly."""
        vendor = "vendor_0"
        product = self.env.vendors_products[vendor][0]

        initial_stock = product.stock
        self.env.update_product_stock(vendor, product, 5)
        self.assertEqual(product.stock, initial_stock + 5)

    def test_vendor_remove_product(self):
        """Test if vendors can remove products correctly."""
        vendor = "vendor_0"
        product = self.env.vendors_products[vendor][0]

        initial_count = len(self.env.vendors_products[vendor])
        self.env.remove_product(vendor, product)

        self.assertEqual(len(self.env.vendors_products[vendor]), initial_count - 1)

        ### 5️⃣ TEST ADMIN ACTIONS ###

    def test_admin_fine_vendors(self):
        """Ensure admin can fine vendors for price gouging."""
        vendor = "vendor_0"
        self.env.fine()

        self.assertIn(vendor, self.env.violations)

    def test_admin_monitor_market(self):
        """Ensure market monitoring runs without errors."""
        self.env.monitor_market()

        ### 6️⃣ TEST REWARD SYSTEM ###

    def test_rewards(self):
        """Check if rewards are correctly assigned."""
        customer = "customer_0"
        vendor = "vendor_0"
        self.env.assign_rewards(customer, 1)

        self.assertGreaterEqual(self.env.rewards[customer], 0)

        ### 7️⃣ TEST TERMINATION CONDITIONS ###

    def test_check_done(self):
        """Ensure the environment stops when conditions are met."""
        self.assertFalse(self.env.check_done())

    def test_termination_logic(self):
        """Ensure termination function works correctly."""
        self.assertTrue(any(agent not in self.env.done_agents for agent in self.env.agents))

    def test_truncation_logic(self):
        """Ensure truncation function works correctly."""
        self.assertFalse(self.env.truncations())

if __name__ == "__main__":
    unittest.main()