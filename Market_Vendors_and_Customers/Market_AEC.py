from platform import processor

from pettingzoo import AECEnv
from pettingzoo.utils import wrappers
import numpy as np
import gymnasium
from gymnasium import spaces


########## PRODUCT CLASS ##########
class Product:

    def __init__(self, product_id, price, name, category, stock):
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
        self.agent_selection = self.agents[0] # First agent starts the run
        self.agent_index = 0

        ########### TRACKING CUSTOMER BUDGET AND VENDOR PRODUCTS ##########
        ## 1. CUSTOMER
        self.customer_budgets = {f'customer_{i}': np.random.randint(50, 2000) for i in range(customers)}

        ##2. VENDORS
        self.vendors_products = {f'vendor_{i}': [
            Product(product_id=1, price=np.random.randint(500, 1000), name="Iphone", category="Electronics", stock=20),
            Product(product_id=2, price=np.random.randint(10,30), name="Apple", category="Fruit", stock=100),
            Product(product_id=3, price=np.random.randint(5, 15), name="Coca-Cola", category="Beverages", stock=80),
        ] for i in range (vendors)}

        self.vendor_revenue ={f"vendor_{i}": 0 for i in range(self.vendors)}
        self.rewards = {agent: 0 for agent in self.agents}

        ############  ADMIN SPECIFIC ATTRIBUTES ##############
        self.violations = {}
        self.vendor_penalties = {}
        self.max_price_factor = 1.5
        self.penalty_steps = 3


        ########### ACTION SPACES FOR AGENTS ##########

        self.action_spaces = {
            agent: spaces.Discrete(6) if "customer" in agent else spaces.Discrete(4)
            for agent in self.agents
         }
        self.action_spaces["admin"] = spaces.Discrete(3)

        admin_actions = {
            0: "enforce_policies",
            1: "fine_vendors",
            2: "monitor_market",

        }
        #Vendor Action Space
        vendor_action_space = {
            0: "adjust_price",
            1: "add_product",
            2: "remove_product",
            3: "update_stock"
        }

        #Customer Action Space
        customer_action_space = {
            0: "browse_products",
            1: "purchase_products",
            2: "negotiate_products",
            3: "wait",
            4: "check_market_trends",
            5: "set_purchase_priority"
        }

        ########### OBSERVATION SPACES ##########
        self.observation_spaces = {}
        for agent in self.agents:
        # ADMIN, CUSTOMERS and VENDORS
            if "admin" in agent:
                self.observation_spaces[agent]= spaces.Dict({
                    "market_trends": spaces.Discrete(3),
                    "vendor_status": spaces.MultiBinary(self.vendors),
                    "price_distributions": spaces.Box(low=0, high=100, shape=(self.vendors,), dtype=np.float32),
                    "total_transactions": spaces.Box(low=0, high=100, shape=(1,), dtype=np.int32),
                    "violations_detected": spaces.Discrete(10)
                })
            elif "customer" in agent:

                self.observation_spaces[agent] = spaces.Dict({
                    "vendors": spaces.MultiBinary(self.vendors),
                    "prices": spaces.Box(low=0, high=100, shape=(self.vendors,), dtype=np.float32),
                    "stock": spaces.Box(low=0, high=100, shape=(self.vendors,), dtype=np.int32),
                    "market_trend": spaces.Discrete(3),
                    "budget": spaces.Box(low=0, high=100, shape=(1,), dtype=np.float32),
                })
            elif "vendor" in agent:
                self.observation_spaces[agent] = spaces.Dict({
                    "competitor_prices": spaces.Box(low=0, high=100, shape=(self.vendors,), dtype=np.float32),
                    "own_stock": spaces.Box(low=0, high=100, shape=(1,), dtype=np.int32),
                    "market_demand": spaces.Discrete(3),
                    "sales_history": spaces.Box(low=0, high=100, shape=(5,), dtype=np.float32),
                    "status": spaces.Discrete(2)
                })
        ############  TRACKING MARKET ##############
        self.market_prices = np.random.randint(10, 50, size=self.vendors)
        self.market_price_history = {f"vendor_{i}": [] for i in range(self.vendors)}
        self.current_actions = {agent: None for agent in self.agents}
        self.done_agents = set()
        self.vendors_status = {f"vendor_{i}": "active" for i in range(self.vendors)}

########### RESET  ##########
    def reset(self):
        self.market_prices = np.random.randint(10, 50, size=self.vendors)
        self.done_agents = set()
        self.agent_selection = self.agents[0]
        self.agent_index = 0
        self.vendors_products = {f"vendor_{i}": [] for i in range(self.vendors)}
        self.vendors_status = {f"vendor_{i}": "active" for i in range(self.vendors)}


########### OBSERVATION  ##########
    def observe(self, agent):
        """Structured observation for customers"""
        if "customer" in agent:
            return {
                "vendors": [int(v) for v in np.random.choice([0, 1], size=self.vendors)],
                "prices": [float(p) for p in self.market_prices.astype(np.float32)],
                "stock": [int(len(self.vendors_products[f'vendor_{i}'])) for i in range(self.vendors)],
                "market_trend": int(np.random.choice([-1, 0, 1])), # -1: decreasing, 0: stable, 1: increasing
                "budget": np.array(np.random.randint(10, 100), dtype=np.float32),
            }
        elif "vendor" in agent:
            return{
                "competitor_prices": [float(p) for p in self.market_prices.astype(np.float32)],
                "own_stock": [int(len(self.vendors_products[agent]))],
                "market_demand": int(np.random.choice([-1, 0, 1])),
                "sales_history": [float(np.random.randint(0, 100)) for _ in range(5)],
                "status": int(self.vendors_status[agent] == "active"),
            }
        return None

#############  STEP  ##########
    def step(self, action):
        """Agent takes action based on observation"""
        agent = self.agent_selection
        self.current_actions[agent] = action

        self.monitor_market() # Updates market conditions

        if "admin" in agent:
            if action == 0:
                self.impose_price_control()
            elif action == 1:
                self.fine_vendors()
            elif action == 2:
                self.adjust_market_trends()

        self.assign_rewards(agent, action)

        self.agent_index = (self.agents.index(agent) + 1) % len(self.agents)
        self.agent_selection = self.agents[self.agent_index]

        if len(self.done_agents) >= len(self.agents):
            self.agents = []

    #########  REWARD FUNCTION ############
    def assign_rewards(self, agent, action):
        """Assign rewards based on agents action and market dynamics"""
        if "customer" in agent:
            if action == 1:
                vendor = self.select_vendor(agent)
                product = self.select_product(vendor)
                if product and self.customer_budgets[agent] >= product.price:
                    self.customer_budgets[agent] -= product.price
                    self.vendor_revenue[vendor] += product.price
                    self.rewards[agent] += 10
                    self.rewards[vendor] += 15
                else:
                    self.rewards[agent] -= 5 #Penalty for not buying

        elif "vendor" in agent:
            if action == 0: #Adjust price
                product = self.select_product(agent)
                if product.price > self.average_market_price * 2:
                    self.rewards[agent] -= 10
                else:
                    self.rewards[agent] += 5

        elif "admin" in agent:
            if action == 2: #Monitor market
                if self.active_vendors >= self.vendors * 0.7:
                    self.rewards[agent] += 10
                else:
                    self.rewards[agent] -= 10



#########  RENDER ############
    def render(self, mode='human'):
        """Prints out  the Market prices"""
        print(f"Market Prices: {self.market_prices}")
        print(f"Vendors Products: {self.vendors_products}")
        print(f"Vendors Status: {self.vendors_status}")

    ############  CLOSE  ##############
    def close(self):
        pass

    ############  ADMIN FUNCTIONALITIES  ##############

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
        obs = self.observe(customer)
        if vendor in self.vendors_products and product in self.vendors_products[vendor]:  # Checks if vendor has a product
            if self.customer_budgets[customer] >= product.price:
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
            else:
                print(f"{customer} does have enough budget to purchase the {product}")

    def negotiate(self, customer, vendor, product, new_price):
        """Customer negotiates price with vendor"""
        obs = self.observe(customer)
        if vendor in self.vendors_products and product in self.vendors_products[vendor]:
            print(f"{customer} negotiated {product} with {vendor}. New price: {new_price}")

    def walk_away(self, customer):
        """Customer decided to walk away"""
        print(f"{customer} walked away")

    def check_market_trend(self, customer):
        """Customer observes market price"""
        print(f"{customer} observes market trends: {obs['market_trends']}")

    def browse_products(self, customer):
        """Customer browses products"""
        obs = self.observe(customer)
        print(f"{customer} is browsing products: {obs['vendors']}")

    ############  VENDOR FUNCTIONALITIES  ##############

    def add_product(self,vendor, product):
        """Adds a product to the market prices"""
        if vendor in self.vendors_products:
            self.vendors_products[vendor].append(product)
            self.update_status(vendor)

    def remove_product(self, vendor, product):
        """Removes a product from the market prices"""
        if vendor in self.vendors_products and product in self.vendors_products[vendor]:
            self.vendors_products[vendor].remove(product)
            print(f"{vendor} sold {product}")
            self.update_status(vendor)

    def sell_product(self, vendor, product):
        if vendor in self.vendors_products and product in self.vendors_products[vendor]:
            self.vendors_products[vendor].remove(product)
            sef.vendor_revenue[vendor] += product.price

    def update_product_stock(self, vendor, product, quantity):
        """Updates stock"""
        if vendor in self.vendors_products and product in self.vendors_products[vendor]:
            index = self.vendors_products[vendor].index(product)
            self.vendors_products[vendor][index].update_stock(quantity)


    def get_product_details(self, vendor):
        """Returns product details"""
        if vendor in self.vendors_products:
            return [product.get_details() for product in self.vendors_products[vendor]]

    def update_status(self, vendor):
        """Update vendor's status"""
        if vendor in self.vendors_status:
            vendor_products = self.vendors_products[vendor]

        #BECOME INACTIVE IF THERE IS NO PRODUCT
        if not vendor_products:
            self.vendors_status[vendor] = "inactive"

        # BECOME INACTIVE IF THERE ARE NO SALES IN A LONG TIME
        elif len(self.market_price_history[vendor]) > 5 and sum(self.market_price_history[vendor][-5:]) == 0:
            self.vendors_status[vendor] = "inactive"

        #ACTIVE IF THEY HAVE PRODUCTS
        else:
            self.vendors_status[vendor] = "active"

        print(f"{vendor} status: {self.vendors_status[vendor]}")

    def analyze_market_trends(self, vendor):
        """Analyze market trends"""
        if vendor not in self.market_price_history:
            return "No data Available"

        history = self.market_price_history[vendor]
        if len(history) < 3:
            return "Insufficient Data"

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
                    product.set_price((product.price + avg_competitior_price) / 2)

                if product.price ==50:
                    product.set_price(50 * 1.1)

                print(f"{vendor} adjusted {product.name}'s price to {product.price}")


        ## Vendor Adjust market
        self.market_price_history[vendor].append(self.market_prices[int(vendor.split("_")[1])])
        if len(self.market_price_history[vendor]) > 10:
            self.market_price_history[vendor].pop(0)








env = wrappers.CaptureStdoutWrapper(Market_AEC_Env())
