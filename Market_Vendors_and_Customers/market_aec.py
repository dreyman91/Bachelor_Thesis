
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
from decimal import Decimal, ROUND_HALF_UP

from pettingzoo.utils.env import AgentID


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

    @staticmethod
    def round_price(value):
        return float(Decimal(value).quantize(Decimal("1.00"), rounding=ROUND_HALF_UP))
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
        MAX_OBS_VALUE = 3000.0

        ####### AGENTS ########
        self.agents = (
            [f"customer_{i}" for i in range(customers)]
            + [f"vendor_{i}" for i in range(vendors)]
        )
        self.possible_agents = self.agents[:]
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.reset()
        self.terminations = {agent: False for agent in self.possible_agents}
        self.truncations = {agent: False for agent in self.possible_agents}
        self._cumulative_rewards = {agent: 0 for agent in self.possible_agents}
        self.infos = {agent: {} for agent in self.agents}
        self.observation_spaces = self._define_observation_spaces()

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
        self.action_spaces = {
            f"customer_{i}": spaces.Discrete(2) for i in range(self.customers)}
        self.action_spaces.update({
            f"vendor_{i}": spaces.Discrete(4) for i in range(self.vendors)})


    def action_space(self, agent):
        """Return  action space for agents"""
        return self.action_spaces[agent]

    def observation_space(self, agent):
        """Return observation space for agents"""
        return self.observation_spaces[agent]

    def _define_observation_spaces(self):
        observation_spaces = {}
        max_obs_size = 1 + self.vendors * 3 + 5

        for agent in self.agents:
            if "customer" in agent:
                obs_space = spaces.Dict(
                    {
                        "vendors": spaces.MultiBinary(self.vendors),
                        "prices": spaces.Box(low=0, high=100, shape=(self.vendors,), dtype=np.float32),
                        "stock": spaces.Box(low=0, high=100, shape=(self.vendors,), dtype=np.int32),
                        "market_trend": spaces.Discrete(3),
                        "budget": spaces.Box(low=0, high=100, shape=(1,), dtype=np.float32),
                    }
                )
            elif "vendor" in agent:
                obs_space = spaces.Dict(
                    {
                        "competitor_prices": spaces.Box(low=0, high=100, shape=(self.vendors,), dtype=np.float32),
                        "own_stock": spaces.Box(low=0, high=100, shape=(1,), dtype=np.int32),
                        "market_demand": spaces.Discrete(3),
                        "sales_history": spaces.Box(low=0, high=100, shape=(5,), dtype=np.float32),
                        "status": spaces.MultiBinary(1),
                    }
                )
            # store raw obs
            observation_spaces[agent] = obs_space

            # convert dict to a flat box space for pettingzoo
        for agent in observation_spaces:
            flattened_size = spaces.utils.flatten_space(observation_spaces[agent]).shape[0]
            max_obs_size = max(max_obs_size, flattened_size)

            observation_spaces[agent] = spaces.Box(
                low=np.zeros(max_obs_size, dtype=np.float32),
                high=np.full(max_obs_size, 10000, dtype=np.float32),
                shape=(max_obs_size,), dtype=np.float32
                )
        self.max_obs_size = max_obs_size
        return observation_spaces

    ########### RESET  ##########
    def reset(self, seed=None, options=None):
        """ Resets the environment for new epsiode"""
        if seed is not None:
            np.random.seed(seed)

        self.done_agents = set()
        self.step_count = 0

        self.agents = [f"customer_{i}" for i in range(np.random.randint(1,3))]
        self.agents += [f"vendor_{i}" for i in range(np.random.randint(1,3))]

        self.possible_agents = self.agents[:]

        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.reset()

        self.infos = {agent: {} for agent in self.possible_agents}



        self.terminations = {agent: False for agent in self.possible_agents}
        self.truncations = {agent: False for agent in self.possible_agents}
        # Reset agent-specific market variables
        self.rewards = {agent: 0 for agent in self.agents}
        self.current_actions = {agent: None for agent in self.agents}

        self.observation_spaces = self._define_observation_spaces()

        for products in self.vendors_products.values():
            for product in products:
                product.stock = np.random.randint(5, 100)
        # Reset market attributes
        self.market_prices = {f"vendor_{i}": np.random.randint(10, 50) for i in range(self.vendors)}
        self.market_price_history = {f"vendor_{i}": [] for i in range(self.vendors)}
        self.vendors_status = {f"vendor_{i}": "active" for i in range(self.vendors)}

        return {agent: self.observe(agent) for agent in self.agents}, {}


        ########### STEP  ##########

    def step(self, action):
        """Agent takes action based on observation"""
        if not self.agents:
            raise RuntimeError("Agent is not available")
        if action is None:
            return None

        if self.agent_selection not in self.agents:
            raise KeyError(f"Agent {self.agent_selection} is no longer active. Active agents: {self.agents}")

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
        self.step_count += 1
        self._cumulative_rewards[agent] += self.rewards[agent]
        # Handle agent turns
        if self.check_done():
            for agent in self.agents:
                if self.check_done[agent]:
                    self.terminations[agent] = True
        # Remove terminated agents from self.agents list
        self._remove_terminated_agents()

        self.infos = {agent: {} for agent in self.agents}

        if self.agents:
            self.agent_selection = self._agent_selector.next()
        else:
            self.agent_selection = None
        # Return updated state
        if agent in self.possible_agents:
            if agent not in self.agents:
                self.terminations[agent] = True
                self.truncations[agent] = False

        if self.agent_selection not in self.agents:
            print(
                f"[ERROR] Invalid agent selection after removal: {self.agent_selection}. Active agents: {self.agents}")
            raise KeyError(f"Agent {self.agent_selection} is no longer active.")

        obs = self.observe(agent)
        reward = self.rewards.get(agent, 0)
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}

        #Remove Terminated agents
        for agent in list(self.infos.keys()):
            if self.terminations.get(agent, False) or self.truncations.get(agent, False):
                del self.infos[agent]
        print(f"[DEBUG] self.infos before returning step(): {self.infos}")
        return obs, reward, self.terminations, self.truncations, self.infos

    ########### OBSERVATION  ##########
    def observe(self, agent):
        """Structured observation for customers"""
        obs_dict = self.get_agent_observation(agent)

        #convert dict to a numpy array
        obs_array = np.concatenate([np.array(v).flatten() for v in obs_dict.values()])
        flat_obs = spaces.utils.flatten(self.observation_spaces[agent], obs_array)

        # pad observation to match max_obs_size
        padded_obs = np.zeros(self.max_obs_size, dtype=np.float32)
        padded_obs[:flat_obs.shape[0]] = flat_obs

        clipped_obs = np.clip(
            padded_obs,
            self.observation_spaces[agent].low.min(),
            self.observation_spaces[agent].high.max()
        )
        # Print debug info before processing action
        print(f"Before Step - Agent: {agent}")
        print(f"Current Infos Keys: {list(self.infos.keys())}")

        # Ensure `agent` exists in `infos`
        if agent not in self.infos:
            raise KeyError(f"Agent {agent} is missing from env.infos! Current infos: {self.infos}")

        return padded_obs
    ########### GET OBSERVATIONS #############
    def get_agent_observation(self, agent):
        """"Placeholder for observations"""
        if agent not in self.agents:
            return {}
        if "customer" in agent:
            return {
                "vendors": np.array([
                        1 if self.vendors_products[v] else 0
                        for v in self.vendors_products]
                ),
                "prices": np.array([(
                            self.round_price(np.mean([p.price for p in self.vendors_products[v]]))
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
                "budget": np.array([self.round_price(self.customer_budgets.get(agent, 0))], dtype=np.float32),
            }
        elif "vendor" in agent:
            return {
                "competitor_prices": np.array([(
                            self.round_price(np.mean([p.price for p in self.vendors_products[v]]))
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


    def _remove_terminated_agents(self):
        """Removes terminated agents"""
        self.agents = [agent for agent in self.agents if not self.terminations.get(agent, False)]
    #########  REWARD FUNCTION ############
    def select_vendor(self, customer):
        vendors_list = [
            v for v in self.vendors_products if self.vendors_products[v]]
        return np.random.choice(vendors_list) if vendors_list else None

    def select_product(self, vendor):
        if vendor and vendor in self.vendors_products and self.vendors_products[vendor]:
            return np.random.choice(self.vendors_products[vendor])
        return None

    def check_done(self):
        """
        Checks if agents are done and updates self.terminations accordingly.
        A customer is done if their budget is ≤ 0.
        A vendor is done if they have no products left.
        The environment is done if all customers or all vendors are done.
        """
        done_customers = {
            customer
            for customer in self.customer_budgets
            if self.customer_budgets[customer] <= 0
        }
        done_vendors = {
            vendor
            for vendor in self.vendors_products
            if not self.vendors_products[vendor]
        }

        print(
            f"[DEBUG] Done Customers: {done_customers} | Done Vendors: {done_vendors}"
        )
        for customer in done_customers:
            self.terminations[customer] = True
        for vendor in done_vendors:
            self.terminations[vendor] = True

        all_customers_done = len(done_customers) == len(self.customer_budgets)
        all_vendors_done = len(done_vendors) == len(self.vendors_products)

        if all_customers_done or all_vendors_done:
            print(f"[DEBUG] All agents are done. Terminating environment.")
            self.done_agents.update(done_customers | done_vendors)
            return True
        return False

    def monitor_market(self):
        """Monitor market trends"""
        SALES_THRESHOLD = 5  # Minimum sales to change trend
        PRICE_CHANGE_THRESHOLD = 0.05 # threshold for sales sensitivity

        if len(self.market_price_history) < 3:
            self.market_trend = 0
        else:
            last_prices = [
                history[-3:]
                for history in self.market_price_history.values()
                if len(history) >= 3
            ]

        price_trends = []
        for history in last_prices:
            last_price = np.array(history[-3:])
            if len(last_price) < 3:
                continue
            price_change = (last_price[-1] - last_price[0]) / last_price[0] # % change in the last 3 steps
            price_trends.append(price_change)


        avg_trend = np.mean(price_trends) if price_trends else 0
        if avg_trend > PRICE_CHANGE_THRESHOLD:
            self.market_trend = 1 #High Demand
        elif avg_trend < -PRICE_CHANGE_THRESHOLD:
            self.market_trend = -1 # Low Demand
        else:
            self.market_trend = 0 # Stable

        total_recent_sales = sum(
            sum(history[-3:]) if len(history)>= 3 else sum(history)
            for history in self.market_price_history.values()
        )
        if total_recent_sales > SALES_THRESHOLD:
            self.market_trend = 1 #Increasing demand
        elif total_recent_sales < SALES_THRESHOLD/2:
            self.market_trend = -1 #Decreasing demand



        self.active_vendors = sum(
            1 for v in self.vendors_status.values() if v == "active"
        )
        self.inactive_vendors = self.vendors - self.active_vendors
        self.total_transactions = sum(
            len(products) for products in self.vendors_products.values()
        )

        return {
            "market_trend": self.market_trend,
            "active_vendors": self.active_vendors,
            "inactive_vendors": self.inactive_vendors,
            "average_market_price": self.average_market_price,
        }
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
            self.customer_budgets[customer] -= self.round_price(final_price)
            self.vendor_revenue[vendor] += self.round_price(final_price)

            index = self.vendors_products[vendor].index(product)
            if self.vendors_products[vendor][index].stock > 1:
                self.vendors_products[vendor][index].update_stock(-1)
            else:
                self.vendors_products[vendor].pop(index)

            self.market_price_history[vendor].append(self.round_price(final_price))
            if len(self.market_price_history[vendor]) > 10:
                self.market_price_history[vendor].pop(0)

            print(f"{customer} successfully purchased {product} from {vendor}")
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
        EPISODE_LOCK_THRESHOLD = 5
        MIN_PRICE = 2
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
            avg_competitor_price = np.mean(competitor_prices) if competitor_prices else product.price
            avg_competitor_price = max(avg_competitor_price, 1.0)
            # Price Threshold lock

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
                product.set_price(max(product.price * 0.7, MIN_PRICE)) # decrease prcie by 30%
                self.rewards[vendor] += 0

          # Threshold
        min_price = max(product.price * 0.65, MIN_PRICE)
        if product.price < min_price:
            self.rewards[vendor] -= 10
            product.set_price(min_price)

        # Vendor Adjust market
        self.market_price_history[vendor].append(self.round_price(product.price))
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