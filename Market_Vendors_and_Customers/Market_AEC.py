from platform import processor

from pettingzoo import AECEnv
from pettingzoo.utils import wrappers
import numpy as np
import gymnasium
from gymnasium import spaces


class Product:

    def __init__(self, product_id, price, name, category, stock):
        self.product_id = product_id
        self.price = price
        self.name = name
        self.category = category
        self.stock = stock

    # Update Stock
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


class Market_AEC_Env(AECEnv):
    """
    A multi agent environment for Market vendors and Customers.
    """
    metadata = {'render_modes': ['human', 'rgb_array'], "name": "market_aec_v0"}

# Initialization
    def __init__(self, vendors =3, customers = 3, render_mode="human"):
        super().__init__()
        self.customers = customers
        self.vendors = vendors
        self.render_mode = render_mode

        self.agents = [f'customer_{i}' for i in range(customers)]+[f'vendor_{i}' for i in range(vendors)]
        self.possible_agents = self.agents[:]
        self.agents_selection = self.agents[0] # First agent starts the run
        self.agent_index = 0

        #Action Space
        self.action_spaces = {agent: spaces.Discrete(10) for agent in self.agents}

        #Observation Space
        self.observation_spaces = {agent: spaces.Box(low=0, high=100, shape=(2,), dtype=np.float32) for agent in self.agents}

        self.market_prices = np.random.randint(10, 50, size=self.vendors)
        self.market_price_history = {f"vendor_{i}": [] for i in range(self.vendors)}
        self.current_actions = {agent: None for agent in self.agents}
        self.done_agents = set()

        #Vendor specific attributes
        self.vendors_products = {f"vendor_{i}": [] for i in range(self.vendors)}
        self.vendors_status = {f"vendor_{i}": "active" for i in range(self.vendors)}

# Reset
    def reset(self):
        self.market_prices = np.random.randint(10, 50, size=self.vendors)
        self.done_agents = set()
        self.agent_selection = self.agents[0]
        self.agent_index = 0
        self.vendors_products = {f"vendor_{i}": [] for i in range(self.vendors)}
        self.vendors_status = {f"vendor_{i}": "active" for i in range(self.vendors)}


# Observation
    def observe(self, agent):
        """Agents take turn setting prices, adding/removing products, or making purchases"""
        if "customer" in agent:
            return np.array([min(self.market_prices), np.random.rand()], dtype=np.float32)
        else:
            return np.array([np.mean(self.market_prices), np.random.rand()], dtype=np.float32)

# Step/Action
    def step(self, action):
        """Agent takes turn  setting the price or,making purchases"""
        agent = self.agent_selection
        self.current_actions[agent] = action
        print(f"{agent} takes actiion {action}")


        self.agent_index = (self.agents.index(agent) + 1) % len(self.agents)
        self.agent_selection = self.agents[self.agent_index]

        if len(self.done_agents) >= len(self.agents):
            self.agents = []
# Render
    def render(self, mode='human'):
        """Prints out  the Market prices"""
        print(f"Market Prices: {self.market_prices}")
        print(f"Vendors Products: {self.vendors_products}")
        print(f"Vendors Status: {self.vendors_status}")

# Close
    def close(self):
        pass

    #Vendor Functionalities

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

    def update_product_stock(self, vendor, product, quantity):
        """Updates stock"""
        if vendor in self.vendors_products and product in self.vendors_products[vendor]:
            index = self.vendors_products[vendor].index(product)
            self.vendors_products[vendor][index].update_stock(quantity)

    def set_product_price(self, vendor, product, price):
        """Sets price"""
        if vendor in self.vendors_products and product in self.vendors_products[vendor]:
            index = self.vendors_products[vendor].index(product)
            self.vendors_products[vendor][index].set_price(price)

    def get_product_details(self, vendor):
        """Returns product details"""
        if vendor in self.vendors_products:
            return [product.get_details() for product in self.vendors_products[vendor]]

    def update_status(self, vendor):
        """Update vendor's status"""
        if vendor in self.vendors_status:
            return

        vendor_products = self.vendors.product[vendor]

        #become inactive if there is no product
        if not vendors_products:
            self.vendors_status[vendor] = "inactive"

        #become inactive after no sales in a long time
        elif len(sales.market_price_history[vendor]) > 5 and sum(self.market_price_history[-5:])== 0:
            self.vendors_status[vendor] = "inactive"

        #Active if they have products and there is demand
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

        print(f"{vendor} observes market trends: {trend}")
        return trend

    def adjust_prices(self, vendor):
        """Adjust vendor prices based on market trends, stock levels, and competitor presence"""
        if vendor in self.vendors_products:
            trend = self.analyze_market_trends(vendor)
            vendor_products = self.vendors_products[vendor]

            for product in vendor_products:
                competitor_prices = [
                    p.price for v, products in self.vendors_products.items() if v != vendor for p in products  if p.name == product.name
                ]
                avg_competitior_price = np.mean(competitor_prices) if competitor_prices else  product.price

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

                #Stay competitve
                if competitor_prices:
                    product.set_price((product.price + avg_competitior_price) / 2)

                print(f"{vendor} adjusted {product.name}'s price to {product.price}")


        ## Vendor Adjust market
        self.market_price_history[vendor].append(self.market_prices[int(vendor.split("_")[1])])
        if len(self.market_price_history[vendor]) > 10:
            self.market_price_history[vendor].pop(0)





    # Customer functionalities

    def purchase(self, customer, vendor, product):
        """Customer purchases the product"""
        if vendor in self.vendors_products and product in self.vendors_products[vendor]:
            index = self.vendors_products[vendor].index(product)
            if self.vendors_products[vendor][index].stock > 1:
                self.vendors_products[vendor][index].update_stock(-1)
            else:
                self.vendors_products[vendor].pop(index)
            print(f"{customer} purchased {product} from {vendor}")

            self.adjust_prices(vendor)
            self.update_status(vendor)

    def negotiate(self, customer, vendor, product, new_price):
        """Customer negotiates price with vendor"""
        if vendor in self.vendors_products and product in self.vendors_products[vendor]:
            print(f"{customer} negotiated {product} with {vendor}. New price: {new_price}")

    def walk_away(self, customer):
        """Customer decided to walk away"""
        print(f"{customer} walked away")

    def observe_market(self, customer):
        """Customer observes market price"""
        print(f"{customer} observes market prices: {self.market_prices}")

    def browse_products(self, customer):
        """Customer browses products"""
        print(f"{customer} is browsing products")


env = wrappers.CaptureStdoutWrapper(Market_AEC_Env())
