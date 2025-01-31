from platform import processor

from pettingzoo import AECEnv
from pettingzoo.utils import wrappers
import numpy as np
import gymnasium
from gymnasium import spaces


########## PRODUCT CLASS ##########
class Product:

    def __init__(self, product_id, price, name, category, stock):
        if price < 0 or stock < 0:
            raise ValueError("price and stock cannot be negative")
        self.product_id = product_id
        self.price = price
        self.name = name
        self.category = category
        self.stock = stock

    def update_stock(self, quantity):
        """Updates stock"""
        self.stock += quantity

    def set_price(self, price):
        """Sets price"""
        self.price = price

    def get_details(self):
        """Returns product details as a string"""
        return f"{self.name} ({self.category}): ${self.price}, Stock: {self.stock}"

    def __repr__(self):
        """Ensures readable string"""
        return self.get_details()

########## MARKET CLASS ##########
class Market_AEC_Env(AECEnv):
    """
    A multi-agent environment for Market vendors and Customers.
    """
    metadata = {'render_modes': ['human', 'rgb_array'], "name": "market_aec_v0"}

########### INITIALIZATION ##########
    def __init__(self, vendors =3, customers = 3, render_mode="human", admin=1):
        super().__init__()
        self.customers = customers
        self.vendors = vendors
        self.admin = admin
        self.render_mode = render_mode
        self.penalty = 5

        ####### AGENTS ########
        self.agents = [f'customer_{i}' for i in range(customers)]+ \
                      [f'vendor_{i}' for i in range(vendors)] + \
                      [f'admin_{i}' for i in range(admin)]

        self.possible_agents = self.agents[:]
        self.agent_index = 0
        self.agent_selection = self.agents[self.agent_index]  # First agent starts the run

        self.customer_budgets = {f'customer_{i}': np.random.randint(50, 2000) for i in range(customers)}

        self.vendors_products = {f'vendor_{i}': [
            Product(product_id=1, price=np.random.randint(500, 1000), name="Iphone", category="Electronics", stock=20),
            Product(product_id=2, price=np.random.randint(10,30), name="Apple", category="Fruit", stock=100),
            Product(product_id=3, price=np.random.randint(5, 15), name="Coca-Cola", category="Beverages", stock=80),
        ] for i in range (vendors)}

        # Market Attributes
        self.vendor_revenue ={f"vendor_{i}": 0 for i in range(self.vendors)}
        self.rewards = {agent: 0 for agent in self.agents}
        self.violations = {}
        self.vendor_penalties = {}
        self.max_price_factor = 1.5
        self.penalty_steps = 3

        # Action and Observation Spaces
        self.action_spaces = {
            agent: spaces.Discrete(6) if "customer" in agent else spaces.Discrete(4)
            for agent in self.agents
         }
        self.action_spaces["admin"] = spaces.Discrete(3)

        ########### OBSERVATIONs ##########
        self.observation_spaces = self._define_observation_spaces()

    def _define_observation_spaces(self):
        observation_spaces = {}
        for agent in self.agents:
            if "admin" in agent:
                observation_spaces[agent]= spaces.Dict({
                    "market_trends": spaces.Discrete(3),
                    "vendor_status": spaces.MultiBinary(self.vendors),
                    "price_distributions": spaces.Box(low=0, high=100, shape=(self.vendors,), dtype=np.float32),
                    "total_transactions": spaces.Box(low=0, high=100, shape=(1,), dtype=np.int32),
                    "violations_detected": spaces.Discrete(10)
                })
            elif "customer" in agent:

                observation_spaces[agent] = spaces.Dict({
                    "vendors": spaces.MultiBinary(self.vendors),
                    "prices": spaces.Box(low=0, high=100, shape=(self.vendors,), dtype=np.float32),
                    "stock": spaces.Box(low=0, high=100, shape=(self.vendors,), dtype=np.int32),
                    "market_trend": spaces.Discrete(3),
                    "budget": spaces.Box(low=0, high=100, shape=(1,), dtype=np.float32),
                })
            elif "vendor" in agent:
                observation_spaces[agent] = spaces.Dict({
                    "competitor_prices": spaces.Box(low=0, high=100, shape=(self.vendors,), dtype=np.float32),
                    "own_stock": spaces.Box(low=0, high=100, shape=(1,), dtype=np.int32),
                    "market_demand": spaces.Discrete(3),
                    "sales_history": spaces.Box(low=0, high=100, shape=(5,), dtype=np.float32),
                    "status": spaces.Discrete(2)
                })

        return observation_spaces

        # ############  TRACKING MARKET ##############
        # self.market_prices = np.random.randint(10, 50, size=self.vendors)
        # self.market_price_history = {f"vendor_{i}": [] for i in range(self.vendors)}
        # self.current_actions = {agent: None for agent in self.agents}
        # self.done_agents = set()
        # self.vendors_status = {f"vendor_{i}": "active" for i in range(self.vendors)}

        ########### Step  ##########

    def step(self, action):
        """Agent takes action based on observation"""
        if not self.agents:
            raise RuntimeError('Agent is not available')
        if action is None:
            return
        agent = self.agent_selection
        self.current_actions[agent] = action

        if "admin" in agent:
            if action == 0:
                self.enforce_policies()
            elif action == 1:
                self.fine()
            elif action == 2:
                self.monitor_market()

        elif "customer" in agent:
            if action == 0:  # Purchase
                vendor = self.select_vendor(agent)
                product = self.select_product(agent)
                if vendor and product and vendor in self.vendors_products:
                    self.purchase(agent, vendor, product)
                else:
                    self.monitor_market()

        elif "vendor" in agent:
            if action == 0:
                self.adjust_prices(agent)
            elif action == 1:
                self.update_product_stock(agent)

        self.assign_rewards(agent, action)

        self.agent_index = (self.agents.index(agent) + 1) % len(self.agents)
        self.agent_selection = self.agents[self.agent_index]

        if self.check_done():
            self.agents = []

########### RESET  ##########
    def reset(self, seed=None, options=None):

        if seed is not None:
            np.random.seed(seed)

        self.done_agents = set()
        self.agent_index = 0
        self.agent_selection = self.agents[self.agent_index]

        self.vendors_products = {f"vendor_{i}": [
            Product(product_id=1, price=np.random.randint(500, 1000), name="Iphone", category="Electronics", stock=20),
            Product(product_id=2, price=np.random.randint(10,30), name="Apple", category="Fruit", stock=100),
            Product(product_id=3, price=np.random.randint(5, 15), name="Coca-Cola", category="Beverages", stock=80),
        ] for i in range(self.vendors)}

        self.rewards = {agent: 0 for agent in self.agents}


        self.market_prices = np.random.randint(10, 50, size=self.vendors)
        self.market_price_history = {f"vendor_{i}": [] for i in range(self.vendors)}
        self.vendors_status = {f"vendor_{i}": "active" for i in range(self.vendors)}
        self.current_actions = {agent: None for agent in self.agents}

        return {agent: self.observe(agent) for agent in self.agents}




########### OBSERVATION  ##########
    def observe(self, agent):
        """Structured observation for customers"""
        if agent not in self.agents:
            raise ValueError(f"Agent {agent} is not in the environment")

        if "admin" in agent:
            return {
                "market_trends": self.market_trend,
                "vendors_status": np.array(
                    [1 if self.vendors_status[v] == "active" else 0 for v in self.vendors_status[agent]]
                ),
                "price_distributions": np.array(
                    [np.mean([p.price for p in  self.vendors_products[v]]) if self.vendors_products[v] else 0
                     for v in self.vendors_products],dtype=np.float32),
                "total_transactions": np.array([self.total_transactions], dtype=np.int32),
                "violations_detected": sum(self.violations.values())
            }

        if "customer" in agent:
            return {
                "vendors": np.array(
                    [1 if self.vendors_products[v] else 0 for v in self.vendors_products]
                ),
                "prices": np.array(
                    [np.mean([p.price for p in self.vendors_products[v]]) if self.vendors_products[v] else 0
                    for v in self.vendors_products], dtype=np.float32),
                "stock": np.array(
                    [sum(p.stock for p in self.vendors_products[v]) if self.vendors_products[v] else 0
                     for v in self.vendors_products], dtype=int32),
                "market_trend": self.market_trend,
                "budget": np.array([self.customer_budgets.get(agent, 0)], dtype=np.float32),
            }
        elif "vendor" in agent:
            return{
                "competitor_prices": np.array(
                    [np.mean([p.price for p in self.vendors_products[v]]) if self.vendors_products[v] else 0
                     for v in self.vendors_products if v!= agent], dtype=np.float32),
                "own_stock": np.array(
                    [sum(p.stock for p in self.vendors_products[agent])], dtype=np.int32),
                "market_demand": self.market_trend,
                "sales_history": np.array(
                    self.market_price_history[agent][-5:] if len(self.market_price_history[agent]) >= 5
                    else self.market_price_history[agent] + [0] * (5 - llen(self.market_price_history[agent])), dtype=np.float32
                ),
                "status": 1 if self.vendors_status[agent] == "active" else 0,
            }
        return None


    #########  REWARD FUNCTION ############
    def select_vendor(self, customer):
        vendors_list = [v for v in self.vendors_products if self.vendors_products[v]]
        return np.random.choice(vendors_list) if vendors_list else None

    def select_product(self, vendor):
        if vendor and vendor in self.vendors_products and self.vendors_products[vendor]:
            return np.random.choice(self.vendors_products[vendor])
        return None

    def assign_rewards(self, agent, action):
        """Assign rewards based on agents action and market dynamics"""
        if agent not in self.rewards:
            self.rewards[agent] = 0

        if "customer" in agent:
            if action == 1:
                vendor = self.select_vendor(agent)
                product = self.select_product(agent)
                if vendor and product and self.customer_budgets[agent] >= product.price:
                    self.rewards[agent] += 10
                    self.rewards[vendor] += 15
                else:
                    self.rewards[agent] -= 5

        elif "vendor" in agent:
            if action == 0: #Adjust price
                product = self.select_product(agent)
                if product and product.price > self.average_market_price * 2:
                    self.rewards[agent] -= 10
                else:
                    self.rewards[agent] += 5

        elif "admin" in agent:
            if action == 2: #Monitor market
                if self.active_vendors >= self.vendors * 0.7:
                    self.rewards[agent] += 10
                else:
                    self.rewards[agent] -= 10

    def check_done(self):
        done_customers = {customer for customer in self.customer_budgets if self.customer_budgets[cudtomer] <= 0}
        done_vendors = {vendor for vendor in self.vendors_products if not self.vendors_products[vendor]}

        if len(done_customers) == len(self.customer_budgets) or len(done_vendors) == len(self.vendors_products):
            return True
        return False

   ########## Terminations ############
    def terminations(selfs):
        return all(agent in self.done_agents for agent in self.agents)

    ########## Truncations ################
    def truncations(self):
        max_steps = 100
        return self.agent_index >= max_steps


    def monitor_market(self):
        if  len(self.market_price_history) < 3:
            self.market_trend = 0
        else:
            last_prices = [history[-3:] for history in self.market_price_history.values()
                           if len(history) >=3]
            avg_trend = np.mean(np.diff(last_prices))
            self.market_trend = np.sign(avg_trend)

        self.active_vendors = sum(1 for v in self.vendors_status.values() if v=="active")
        self.inactive_vendors = self.vendors - self.active_vendors
        self.total_transactions = sum(len(products) for products in self.vendors_products.values())
        self.violations = {vendor: self.violations.get(vendor, 0) for vendor in self.vendors_products}
        violation_count = sum(1 for vendor in self.violations if self.violations[vendor] > 0)
        print(
            f"Active Vendors: {self.active_vendors}, Market Trend: {self.market_trend}, Violations: {violation_count}")

        avg_prices = [np.mean([p.price for p in self.vendors_products[vendor]])
                               for vendor in self.vendors_products if self.vendors_products[vendor]]
        self.average_market_price = np.mean(avg_prices) if avg_prices else 0

        return {
            "market_trend": self.market_trend,
            "active_vendors": self.active_vendors,
            "inactive_vendors": self.inactive_vendors,
            "average_market_price": self.average_market_price,
        }




    def fine(self):
        for vendor, products in self.vendors_products.items():
            for product in products:
                avg_market_price = np.mean(self.market_prices)
                price_threshold = avg_market_price * self.max_price_factor

                if product.price > price_threshold:
                    self.violations[vendor] = self.violations.get(vendor, 0) + 1
                    self.vendor_penalties[vendor] = self.penalty
                    self.vendors_status[vendor] = self.penalty
                    print(f"Admin penalized Vendor: {vendor}) for price gouging on {product}")

    def update_vendor_status(self):
        """ Reduce penalized vendor for 3 steps"""
        for vendor in self.vendor_penalties:
            if self.vendor_penalties[vendor] > 0:
                self.vendor_penalties[vendor] -= 1
                if self.vendor_penalties[vendor] == 0:
                    self.vendors_status[vendor] = "active"
                    print(f"{vendor} is active again after penalty")

    def enforce_policies(self):
        """Apply regulatory actions based on market conditions"""
        self.fine()
        self.update_vendor_status()

    ############  CUSTOMER FUNCTIONALITIES  ##############

    def purchase(self, customer, vendor, product):
        """Customer purchases the product"""
        if vendor in self.vendors_products and product in self.vendors_products[vendor]:  # Checks if vendor has a product
            if self.customer_budgets[customer] >= product.price:
                if not product in self.vendors_products[vendor]:
                    return
                index = self.vendors_products[vendor].index(product)

                self.customer_budgets[customer] -= product.price
                self.vendor_revenue[vendor] += product.price

                if self.vendors_products[vendor][index].stock > 1:
                    self.vendors_products[vendor][index].update_stock(-1)
                else:
                    self.vendors_products[vendor].pop(index)
                print(f"{customer} purchased {product} from {vendor}")

                self.adjust_prices(vendor)
                self.update_status(vendor)

                customer_action = self.current_actions.get(customer, None)
                vendor_action = self.current_actions.get(vendor, None)
                self.assign_rewards(customer, customer_action)
                self.assign_rewards(vendor, vendor_action)

            else:
                print(f"{customer} does not have enough budget to purchase the {product}")

    def check_market_trend(self, customer):
        """Customer observes market price"""
        print(f"{customer} observes market trends: {obs['market_trends']}")

    def browse_products(self, customer):
        """Customer browses products"""
        obs = self.observe(customer)
        print(f"{customer} is browsing products: {obs['vendors']}")

    def add_product(self,vendor, product):
        """Adds a product to the market prices"""
        existing_product = next((p for p in self.vendors_products[vendor] if p.name==product.name), None)
        if existing_product:
            existing_product.update_stock(product.stock)
        else:

            self.vendors_products[vendor].append(product)
            self.update_status(vendor)

    def remove_product(self, vendor, product):
        """Removes a product from the market prices"""
        if not self.vendors_products[vendor]:
            self.vendors_status[vendor] = "inactive"

        elif vendor in self.vendors_products and product in self.vendors_products[vendor]:
            self.vendors_products[vendor].remove(product)
            print(f"{vendor} sold {product}")
            self.update_status(vendor)


    def update_product_stock(self, vendor, product, quantity):
        """Updates stock"""
        if vendor in self.vendors_products and product in self.vendors_products[vendor]:
            index = self.vendors_products[vendor].index(product)
            self.vendors_products[vendor][index].update_stock(quantity)

    def update_status(self, vendor):
        """Update vendor's status"""
        if vendor not in self.vendors_status:
            vendor_products = self.vendors_products[vendor]
        if not vendor_products:
            print("{vendor} does not exist" )
            return
        vendor_products = self.vendors_products.get(vendor, [])

        if not vendor_products:
            self.vendors_status[vendor] = "inactive"
        elif len(self.market_price_history[vendor]) > 5 and sum(self.market_price_history[vendor][-5:]) == 0:
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
            return None
        last_three = history[-3:]
        avg_change = (last_three[-1] - last_three[0]) / 3

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
        if vendor in self.vendors_products:
            trend = self.analyze_market_trends(vendor)
            vendor_products = self.vendors_products[vendor]
            if vendor not in self.market_price_history:
                self.market_price_history[vendor] = []

            for product in vendor_products:
                competitor_prices = [
                    p.price for v, products in self.vendors_products.items() if v != vendor for p in products  if p.name == product.name
                ]
                avg_competitor_price = np.mean(competitor_prices) if competitor_prices else  product.price
                if np.isnan(avg_competitor_price):
                    avg_competitor_price = product.price

                #Stock based

                if product.stock < 5:
                    product.set_price(product.price * 1.2) # increase price by 20%
                    print(f"{vendor} increased {product.name}'s price to {product.price}")
                elif product.stock > 20:
                    product.set_price(product.price * 0.9) # decrease price by 10%

                #Trend based
                if trend == "increasing":
                    product.set_price(product.price * 1.05)
                elif trend == "decreasing":
                    product.set_price(product.price * 0.95)

                #Stay competitive
                if competitor_prices:
                    product.set_price((product.price + avg_competitor_price) / 2)

                if product.price ==50:
                    product.set_price(50 * 1.1)

                print(f"{vendor} adjusted {product.name}'s price to {product.price}")


        ## Vendor Adjust market
        self.market_price_history[vendor].append(self.market_prices[int(vendor.split("_")[1])])
        if len(self.market_price_history[vendor]) > 10:
            self.market_price_history[vendor].pop(0)

    #########  RENDER ############
    def render(self, mode='human'):
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


env = wrappers.CaptureStdoutWrapper(Market_AEC_Env())
