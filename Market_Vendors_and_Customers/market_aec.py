
r"""

Market_AEC_Env: A multi-agent reinforcement learning environment \n
for simulating a market with customers, vendors, and administrators.

Author: Oluwadamilare Adegun
Date: 10.01.2025.
"""
from operator import index
from platform import processor
from pettingzoo.utils import agent_selector
from pettingzoo import AECEnv
from pettingzoo.utils import wrappers
import numpy as np
import gymnasium
from gymnasium import spaces

########## PRODUCT CLASS ##########


class Product:
    """Product Class."""

    def __init__(self, product_id, price, name, category, stock):
        if price < 0 or stock < 0:
            raise ValueError("price and stock cannot be negative")
        self.product_id = product_id
        self.price = price
        self.name = name
        self.category = category
        self.stock = stock

    def update_stock(self, quantity):
        """Update stock."""
        self.stock += quantity

    def set_price(self, price):
        """Set price."""
        self.price = price

    def get_details(self):
        """Returns product details as a string."""
        return f"{self.name} ({self.category}): ${self.price}, Stock: {self.stock}"

    def __repr__(self):
        """Ensures readable string"""
        return self.get_details()


########## MARKET CLASS ##########
class MarketAECEnv(AECEnv):
    """
    A multi-agent environment for Market vendors and Customers.
    """

    metadata = {
        "render_modes": ["human","rgb_array"],"name": "market_aec_v0"}

    ########### INITIALIZATION ##########
    def __init__(self, vendors=3, customers=3, render_mode="human"):
        """"Initialization"""
        super().__init__()
        self.vendors = vendors
        self.customers = customers
        self.render_mode = render_mode
        self.market_trend = 0
        self.total_transactions = 0
        self.step_count = 0
        self.done_agents = set()
        self.action_spaces = {}

        ####### AGENTS ########
        self.agents = (
            [f"customer_{i}" for i in range(customers)]
            + [f"vendor_{i}" for i in range(vendors)]
        )

        self.possible_agents = self.agents[:]
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.reset()

        ####### CUSTOMER ATTRIBUTES ########
        self.customer_budgets = {
            f"customer_{i}": np.random.randint(50, 2000) for i in range(customers)
        }
        ####### VENDOR ATTRIBUTES ########
        self.vendors_products = {
            f"vendor_{i}": [
                Product(
                    product_id=1,
                    price=np.random.randint(500, 1000),
                    name="Iphone",
                    category="Electronics",
                    stock=20,
                ),
                Product(
                    product_id=2,
                    price=np.random.randint(10, 30),
                    name="Apple",
                    category="Fruit",
                    stock=100,
                ),
                Product(
                    product_id=3,
                    price=np.random.randint(5, 15),
                    name="Coca-Cola",
                    category="Beverages",
                    stock=80,
                ),
            ]
            for i in range(vendors)
        }
        self.vendor_revenue = {f"vendor_{i}": 0 for i in range(self.vendors)}
        self.vendors_status = {f"vendor_{i}": "active" for i in range(self.vendors)}

        ############# MARKET ATTRIBUTES ############
        self.market_prices = {f"vendor_{i}": np.random.randint(10, 50) for i in range(self.vendors)}
        self.market_price_history = {f"vendor_{i}": [] for i in range(self.vendors)}

        ############ MARKET REGULATIONS ######################
        self.rewards = {agent: 0 for agent in self.agents}
        self.current_actions = {agent: None for agent in self.agents}
        self.max_price_factor = 1.5
        self.penalty_steps = 3


        # Action and Observation Spaces
        self.action_spaces["customers"] = spaces.Discrete(2)
        self.action_spaces["vendors"] = spaces.Discrete(4)
        self.observation_spaces = self._define_observation_spaces()

    def _define_observation_spaces(self):
        observation_spaces = {}
        for agent in self.agents:
            if "customer" in agent:

                observation_spaces[agent] = spaces.Dict(
                    {
                        "vendors": spaces.MultiBinary(self.vendors),"prices": spaces.Box(
                        low=0, high=100, shape=(self.vendors,), dtype=np.float32),
                        "stock": spaces.Box(
                            low=0, high=100, shape=(self.vendors,), dtype=np.int32),
                        "market_trend": spaces.Discrete(3),
                        "budget": spaces.Box(low=0, high=100, shape=(1,), dtype=np.float32),
                    }
                )
            elif "vendor" in agent:
                observation_spaces[agent] = spaces.Dict(
                    {
                        "competitor_prices": spaces.Box(
                            low=0, high=100, shape=(self.vendors,), dtype=np.float32),
                        "own_stock": spaces.Box(low=0, high=100, shape=(1,), dtype=np.int32),
                        "market_demand": spaces.Discrete(3),
                        "sales_history": spaces.Box(low=0, high=100, shape=(5,), dtype=np.float32),
                        "status": spaces.MultiBinary(1),
                    }
                )

        return observation_spaces

    ########### RESET  ##########
    def reset(self, seed=None, options=None):
        """ Resets the environment for new epsiode"""
        if seed is not None:
            np.random.seed(seed)

        self.done_agents = set()
        self.step_count = 0
        self.agent_selection = self._agent_selector.reset()

        for products in self.vendors_products.values():
            for product in products:
                product.stock = np.random.randint(5, 100)

        # Reset market attributes
        self.market_prices = {f"vendor_{i}": np.random.randint(10, 50) for i in range(self.vendors)}
        self.market_price_history = {f"vendor_{i}": [] for i in range(self.vendors)}
        self.vendors_status = {f"vendor_{i}": "active" for i in range(self.vendors)}

        # Reset agent-specific market variables
        self.rewards = {agent: 0 for agent in self.agents}
        self.current_actions = {agent: None for agent in self.agents}
        return {agent: self.observe(agent) for agent in self.agents}, {}


        ########### STEP  ##########

    def step(self, action):
        """Agent takes action based on observation"""
        if not self.agents:
            raise RuntimeError("Agent is not available")
        if action is None:
            return None

        # Select current agent
        agent = self.agent_selection
        self.current_actions[agent] = action

        if "customer" in agent:
            if action == 0:  # Purchase
                vendor = self.select_vendor(agent)
                product = self.select_product(agent)

                if vendor and product:
                    last_price = (
                        self.market_price_history[vendor][-1]
                        if self.market_price_history[vendor][-1]
                        else self.market_prices
                    )
                    if product.price > last_price * 1.2:
                        print(f"{agent} is avoiding {vendor} due to high price")
                        self.rewards[agent] -= 3
                    else:
                        self.purchase(agent, vendor, product)

            elif action == 1: # browsing market
                best_vendor = min(
                    self.vendors_products,
                    key=lambda v: np.mean([p.price for p in self.vendors_products[v]]),
                )
                print(f"{agent} is comparing vendors and prefers {best_vendor}")
                self.rewards[agent] += 2

            elif action == 2: # Negotiate
                vendor = self.select_vendor(agent)
                product = self.select_product(agent)
                if vendor and product:
                    self.negotiate(agent, product)

        elif "vendor" in agent:
            if action == 0:
                self.adjust_prices(agent)
            elif action == 1:
                self.update_product_stock(agent)

        # Update environment
        self.assign_rewards(agent, action)
        self.step_count += 1

        # Handle agent turns
        if self.check_done():
            self.done_agents.add(agent)
        while self.agent_selection in self.done_agents:
            self.agent_selection = self._agent_selector.next()

        # Return updated state
        terminations = self.terminations()
        truncations = self.truncations()
        info = {}

        obs = self.observe(agent)
        reward = self.rewards.get(agent, 0)

        return obs, reward, terminations, truncations, info

    ########### OBSERVATION  ##########
    def observe(self, agent):
        """Structured observation for customers"""
        if agent not in self.agents:
            return {}
        if "customer" in agent:
            return {
                "vendors": np.array([
                        1 if self.vendors_products[v] else 0
                        for v in self.vendors_products]
                ),
                "prices": np.array([(
                            np.mean([p.price for p in self.vendors_products[v]])
                            if self.vendors_products[v]
                            else 0)
                        for v in self.vendors_products],
                    dtype=np.float32,
                ),
                "stock": np.array([(
                            sum(p.stock for p in self.vendors_products[v])
                            if self.vendors_products[v]
                            else 0)
                        for v in self.vendors_products],
                    dtype=np.int32,
                ),
                "market_trend": self.market_trend,
                "budget": np.array([self.customer_budgets.get(agent, 0)], dtype=np.float32),
            }
        elif "vendor" in agent:
            return {
                "competitor_prices": np.array([(
                            np.mean([p.price for p in self.vendors_products[v]])
                            if self.vendors_products[v]
                            else 0)
                        for v in self.vendors_products
                        if v != agent],
                    dtype=np.float32,
                ),
                "own_stock": np.array(
                    [sum(p.stock for p in self.vendors_products[agent])], dtype=np.int32
                ),
                "market_demand": self.market_trend,
                "sales_history": np.array((
                        self.market_price_history[agent][-5:]
                        if len(self.market_price_history[agent]) >= 5
                        else self.market_price_history[agent] + [0] * (5 - len(self.market_price_history[agent]))),
                    dtype=np.float32,
                ),
                "status": 1 if self.vendors_status[agent] == "active" else 0,
            }
        return {}

    #########  REWARD FUNCTION ############
    def select_vendor(self, customer):
        vendors_list = [
            v for v in self.vendors_products if self.vendors_products[v]]
        return np.random.choice(vendors_list) if vendors_list else None

    def select_product(self, vendor):
        if vendor and vendor in self.vendors_products and self.vendors_products[vendor]:
            return np.random.choice(self.vendors_products[vendor])
        return None

    ########## Terminations ############
    def terminations(self):
        return {agent: agent in self.done_agents for agent in self.possible_agents}

    ########## Truncations ################
    def truncations(self):
        return {agent: self.step_count >= 100 for agent in self.agents}

    def monitor_market(self):
        if len(self.market_price_history) < 3:
            self.market_trend = 0
        else:
            last_prices = [
                history[-3:]
                for history in self.market_price_history.values()
                if len(history) >= 3
            ]
            avg_trend = np.mean(np.diff(last_prices))
            self.market_trend = np.sign(avg_trend)

        self.active_vendors = sum(
            1 for v in self.vendors_status.values() if v == "active"
        )
        self.inactive_vendors = self.vendors - self.active_vendors
        self.total_transactions = sum(
            len(products) for products in self.vendors_products.values()
        )
        self.violations = {
            vendor: self.violations.get(vendor, 0) for vendor in self.vendors_products
        }
        violation_count = sum(
            1 for vendor in self.violations if self.violations[vendor] > 0
        )
        print(
            f"Active Vendors: {self.active_vendors},"
            f" Market Trend: {self.market_trend},"
            f" Violations: {violation_count}"
        )

        avg_prices = [
            np.mean([p.price for p in self.vendors_products[vendor]])
            for vendor in self.vendors_products
            if self.vendors_products[vendor]
        ]
        self.average_market_price = np.mean(avg_prices) if avg_prices else 0

        return {
            "market_trend": self.market_trend,
            "active_vendors": self.active_vendors,
            "inactive_vendors": self.inactive_vendors,
            "average_market_price": self.average_market_price,
        }


    def enforce_policies(self):
        """Apply regulatory actions based on market conditions"""
        self.fine()
        self.update_vendor_status()

    ############  CUSTOMER FUNCTIONALITIES  ##############

    def purchase(self, customer, vendor, product):
        """Customer purchases the product"""

        final_price = self.negotiate(vendor, customer, product)

        if final_price is None: #Negotiation fails
            print(f"{customer} cannot afford {product}")
            self.rewards[customer] -= 5
            self.rewards[vendor] += 1
            return


        if self.customer_budgets[customer] >= final_price:
            self.customer_budgets[customer] -= final_price
            self.vendor_revenue[vendor] += final_price

            index = self.vendors_products[vendor].index(product)
            if self.vendors_products[vendor][index].stock > 1:
                self.vendors_products[vendor][index].update_stock(-1)
            else:
                self.vendors_products[vendor].pop(index)
            print(f"{customer} successfully purchased {product} on {vendor}")
            self.rewards[customer] += 5
            self.rewards[vendor] += 5

            self.adjust_prices(vendor)
            self.update_status(vendor)

    def check_market_trend(self, customer):
        """Customer observes market price"""
        obs = self.observe(customer)
        print(f"{customer} observes market trends: {obs['market_trend']}")


    def update_product_stock(self, vendor, product, quantity):
        """Updates stock"""
        if product in self.vendors_products[vendor]:
            product.update_stock(quantity)

    def update_status(self, vendor):
        """Update vendor's status"""
        if vendor not in self.vendors_status:
            print(f"{vendor} not in the the market")
            return

        vendor_products = self.vendors_products.get(vendor, [])

        if not vendor_products:
            self.vendors_status[vendor] = "inactive"
        elif (
            vendor in self.market_price_history
            and len(self.market_price_history[vendor]) > 5
        ):
            if sum(self.market_price_history[vendor][-5:]) == 0:
                self.vendors_status[vendor] = "inactive"
            else:
                self.vendors_status[vendor] = "active"

        print(f"{vendor} status: {self.vendors_status[vendor]}")

    def analyze_market_trends(self, vendor):
        """Analyze market trends"""
        if vendor not in self.market_price_history:
            return None

        history = self.market_price_history[vendor]
        if len(history) < 3:
            return "Stable"
        avg_change = (history[-1] - history[0]) / len(history)
        if avg_change > 1:
            trend = "increasing"
        elif avg_change < -1:
            trend = "decreasing"
        else:
            trend = "stable"

        print(f"{vendor} observes market trends: {trend}")
        return trend

    def adjust_prices(self, vendor):
        """Adjust vendor prices based on market trends, stock levels, and competitor presence"""

        # Check if vendor exists
        if vendor not in self.vendors_products:
            return

        trend = self.analyze_market_trends(vendor) # Market trend
        vendor_products = self.vendors_products[vendor]

        #Validate vendor price history
        if vendor not in self.market_price_history:
            self.market_price_history[vendor] = []

        for product in vendor_products:

            # Competitor prices
            competitor_prices = [
                p.price
                for v, products in self.vendors_products.items()
                if v != vendor
                for p in products
                if p.name == product.name
            ]
            avg_competitor_price = (
                np.mean(
                    competitor_prices) if competitor_prices else product.price
            )
            if np.isnan(avg_competitor_price):
                avg_competitor_price = product.price

            if avg_competitor_price <= product.price * 0.9:
                pass
            elif avg_competitor_price > product.price * 1.2:
                product.set_price(product.price * 0.8) # Reduce price by 20%
                self.rewards[vendor] += 10


            # Stock based

            if product.stock < 5 and trend == "increasing":
                product.set_price(product.price * 1.2)  # increase price by 20%
                self.rewards[vendor] += 3
            elif product.stock > 20 and trend == "increasing":
                product.set_price( product.price * 0.9)  # decrease price by 10%
                self.rewards[vendor] += 1
            elif product.stock > 20 and trend == "decreasing":
                product.set_price( product.price * 0.7) # decrease prcie by 30%
                self.rewards[vendor] += 0

          # Threshold
        min_price = product.price * 0.65
        if product.price < min_price:
            self.rewards[vendor] -= 10
            product.set_price(min_price)

        # Vendor Adjust market
        self.market_price_history[vendor].append(self.market_prices[vendor])
        if len(self.market_price_history[vendor]) > 10:
            self.market_price_history[vendor].pop(0)

    def negotiate(self, vendor, customer, product):
        """Negotiation between Customers and Vendors"""
        if vendor not in self.market_price_history: #Check for vendors
            self.market_price_history[vendor] = [0]
        # vendor_sales = sum(self.market_price_history[vendor])
        # print(f"{vendor} sales history: {vendor_sales}")
        # return  vendor_sales

        demand_trend = self.market_trend # -1(low), 0 (stable), 1(high)
        vendor_sales = sum(self.market_price_history[vendor])
        customer_budget = self.customer_budgets[customer]

        original_price = product.price
        final_price = original_price
        discount_applied = False

        # RULES
        if demand_trend == 1 and  customer_budget < original_price: # High demand; No discount
            final_price = original_price
            self.rewards[vendor] += 5
            self.rewards[customer] -= 3

        elif demand_trend == -1 or vendor_sales < 5 :
            final_price = original_price * 0.85 # 15% discount
            discount_applied = True
            self.rewards[vendor] -= 2
            if customer_budget >= final_price:
                self.rewards[customer] += 3
            else:
                self.rewards[customer] -= 2

        else:
            final_price = original_price
            self.rewards[vendor] += 2
            if customer_budget >= final_price:
                self.rewards[customer] += 3
            else:
                self.rewards[customer] += 1
        return final_price

    #########  RENDER ############
    def render(self, mode="human"):
        print(f"Current agent: {self.agent_selection}")
        print(f"Market Prices: {self.market_prices}")
        print(f"Vendors Status: {self.vendors_status}")
        print(f"Customer Budgets: {self.customer_budgets}")
        print(f"Current Actions: {self.current_actions}")
        print(f"Total Transactions: {self.total_transactions}")
        print("===============================================")

    ############  CLOSE  ##############
    def close(self):
        print("Market environment closed.")


