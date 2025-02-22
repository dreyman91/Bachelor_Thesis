
r"""

Market_AEC_Env: A multi-agent reinforcement learning environment \n
for simulating a market with customers, vendors, and administrators.

Author: Oluwadamilare Adegun
Date: 10.01.2025.
"""
from operator import index
from platform import processor
from pettingzoo.utils import agent_selector
from pettingzoo import ParallelEnv
from pettingzoo.utils import wrappers
import numpy as np
import gymnasium
from gymnasium import spaces
from decimal import Decimal, ROUND_HALF_UP

from pettingzoo.utils.env import AgentID

from Custom_API.AEC_API.AEC_Env_Sample import reward


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
class MarketAECEnv(ParallelEnv):
    """
    A multi-agent environment for Market vendors and Customers.
    """

    @staticmethod
    def round_price(value):
        return float(Decimal(value).quantize(Decimal("1.00"), rounding=ROUND_HALF_UP))

    def format_agent_name(self, role: str, index: int):
        """
        Generates a standardized agent name.
        role: customer or vendor
        index: AgentID
        """
        if role not in ["customer", "vendor"]:
            raise ValueError(f"Role {role} is not supported. Must be customer or vendor.")
        return f"{role}_{index}"
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
        self.action_spaces = {}
        MAX_OBS_VALUE = 3000.0

        ####### AGENTS ########
        self.agents = [self.format_agent_name(role, i) for role in ["customer", "vendor"]
                       for i in range(customers if role == "customer" else vendors)]
        self.possible_agents = self.agents[:]

        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.rewards = {agent: 0 for agent in self.agents}
        self.current_actions = {agent: None for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}

        # properly calling define_observation
        if hasattr(self, "_define_observation_spaces"):
            self.observation_spaces = self._define_observation_spaces
        else:
            raise AttributeError("[ERROR] Please define observation spaces.")
        self.observation_spaces = self._define_observation_spaces()

        ####### CUSTOMER ATTRIBUTES ########
        self.customer_budgets = {
            self.format_agent_name("customer", i): np.random.randint(50, 2000) for i in range(customers)
        }
        ####### VENDOR ATTRIBUTES ########
        self.vendors_products = {
            self.format_agent_name("vendor", i): [
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
        self.vendor_revenue = {self.format_agent_name("vendor", i): 0 for i in range(self.vendors)}
        self.vendors_status = {self.format_agent_name("vendor", i): "active" for i in range(self.vendors)}

        ############# MARKET ATTRIBUTES ############
        self.market_prices = {self.format_agent_name("vendor", i): np.random.randint(10, 50) for i in range(self.vendors)}
        self.market_price_history = {self.format_agent_name("vendor", i): [] for i in range(self.vendors)}

        # Action and Observation Spaces
        self.action_spaces = {
            agent: spaces.Discrete(2) if "customer" in agent else spaces.Discrete(4) for agent in self.agents}

        print("Action Spaces Dictionary:", self.action_spaces)

    def seed(self, seed=None):
        """"Ensures reproductibility by setting a random seed"""
        if seed is not None:
            np.random.seed(seed)


    ########### RESET  ##########
    def reset(self, seed=None, options=None):
        """ Resets the environment for new epsiode"""
        if seed is not None:
            np.random.seed(seed)

        self.step_count = 0

        self.agents = [self.format_agent_name("customer", i) for i in range(self.customers)]
        self.agents += [self.format_agent_name("vendor", i) for i in range(self.vendors)]

        self.possible_agents = self.agents[:] # Reset possible agents
        # Reset dictionaries
        self.infos = {agent: {} for agent in self.agents}
        self.terminations = {agent: False for agent in self.possible_agents}
        self.truncations = {agent: False for agent in self.possible_agents}
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.possible_agents}
        self.current_actions = {agent: None for agent in self.agents}

        # Action Space to match new agents ID
        self.action_spaces = {agent: spaces.Discrete(2) for agent in self.agents if  agent.startswith("customer")}
        self.action_spaces.update({agent: spaces.Discrete(4) for  agent in self.agents if agent.startswith("vendor")})

        obs_shape = (15,)  # Replace with your actual observation shape
        self.observation_spaces = {
            agent: spaces.Box(low=0, high=1, shape=obs_shape, dtype=np.float32) for agent in self.possible_agents
        }


        # Making sure sel.vendors_products includes all vendors
        self.vendors_products = {
            agent: [
                Product(1, np.random.randint(500, 1000), "Iphone", "Electronics", 20),
                Product(2, np.random.randint(10, 30), "Apple", "Fruit", 100),
                Product(3, np.random.randint(5, 15), "Coca-Cola", "Beverages", 80),
            ]
            for agent in self.agents if agent.startswith("vendor")  #Ensure only vendors are included
        }
        # Reset market attributes
        self.market_prices = {self.format_agent_name("vendor", i): np.random.randint(10, 50) for i in range(self.vendors)}
        self.market_price_history = {agent: [] for agent in self.agents if agent.startswith("vendor")}
        self.vendors_status = {agent: "active" for agent in self.agents if agent.startswith("vendor")}

        observations = {agent: self.observe(agent) for agent in self.agents}

        return observations, self.infos

    def _define_observation_spaces(self):
        observation_spaces = {}
        max_obs_size = 0 # Compute maximum obs space based on the largest observation space

        for agent in self.possible_agents: # iterating over agants that could exist in the env
            if agent.startswith("customer"):
                obs_space = spaces.Dict(
                    {
                        "vendors": spaces.MultiBinary(self.vendors),
                        "prices": spaces.Box(low=0, high=500, shape=(self.vendors,), dtype=np.float32),
                        "stock": spaces.Box(low=0, high=2000, shape=(self.vendors,), dtype=np.int32),
                        "market_trend": spaces.Discrete(3),
                        "budget": spaces.Box(low=0, high=500, shape=(1,), dtype=np.float32),
                    }
                )
            elif agent.startswith("vendor"):
                obs_space = spaces.Dict(
                    {
                        "competitor_prices": spaces.Box(low=0, high=500, shape=(self.vendors,), dtype=np.float32),
                        "own_stock": spaces.Box(low=0, high=2000, shape=(1,), dtype=np.int32),
                        "market_demand": spaces.Discrete(3),
                        "sales_history": spaces.Box(low=0, high=100, shape=(5,), dtype=np.float32),
                        "status": spaces.MultiBinary(1),
                    }
                )
            else:
                continue #skip unknown agents

            # store raw obs
            observation_spaces[agent] = obs_space

        #Determine max observation size dynamically
        flattened_size = spaces.utils.flatten_space(observation_spaces[agent]).shape[0]
        max_obs_size = max(max_obs_size, flattened_size)

        print(f"[DEBUG] {agent} Raw Observation Space Size: {flattened_size}")
        # convert dict to a flat box space for for parallel processing
        for agent in observation_spaces:
            observation_spaces[agent] = spaces.Box(
                low=np.zeros(max_obs_size, dtype=np.float32),
                high=np.full(max_obs_size, 1.0, dtype=np.float32),
                shape=(max_obs_size,), dtype=np.float32)

        self.max_obs_size = max_obs_size
        return observation_spaces

    def action_space(self, agent):
        """Return  action space for agents"""
        if agent not in self.action_spaces:
            return self.action_spaces[agent]


        return spaces.Discrete(3) # Return a default space if agent is missing

    def observation_space(self, agent):
        """Return observation space for agents"""
        if agent not in self.observation_spaces:
            return self.observation_spaces[agent]
        return spaces.Box(low=0, high=1, shape=(self.max_obs_size,), dtype=np.float32) # Return default Obs space if agent is missing
        ########### STEP  ##########
    def step(self, actions):
        """Agent takes action based on observation"""
        if not self.agents:
            raise RuntimeError("Agent is not available")

        if not isinstance(actions, dict):
            raise ValueError("Action must be a dictionary")


        # Process each agent action
        for agent, agent_action in self.agents:
            if agent not in self.agents:
                continue # skip removed agents

            self.current_actions[agent] = agent_action

            # Customer Actions
            if agent.startswith("customer"):
                if agent_action == 0:  # Purchase
                    vendor = self.select_vendor(agent)
                    product = self.select_product(agent)

                    # Browses through the market
                    if vendor and product:
                        last_price = self.market_price_history.get(vendor, [self.market_prices[vendor]])[-1]
                        if product.price > last_price * 1.2:
                            print(f"{agent} is avoiding {vendor} due to high price")
                            self.rewards[agent] -= 3
                        else:
                            self.purchase(agent, vendor, product)

                elif agent_action == 1: # browsing market
                    best_vendor = min(
                        self.vendors_products,
                        key=lambda v: np.mean([p.price for p in self.vendors_products[v]]),
                    )
                    print(f"{agent} is comparing vendors and prefers {best_vendor}")
                    self.rewards[agent] += 2

                elif agent_action == 2: # Negotiate
                    vendor = self.select_vendor(agent)
                    product = self.select_product(agent)
                    if vendor and product:
                        self.negotiate(agent, product)

            elif agent.startswith("vendor"):
                if agent_action == 0:
                    self.adjust_prices(agent) #Adjust prices
                elif agent_action == 1:
                    product = self.select_product(agent) # Select a product to updatequantity = np.random.randint(1, 5) # Random quantity to update
                    if product:
                        self.update_product_stock(agent, product, np.random.randint(1,5))

        # Update environment
        self.step_count += 1
        self._cumulative_rewards[agent] += self.rewards[agent]

        # Remove terminated agents
        agents_to_remove = [a for a in self.agents if self.terminations.get(a, False) or self.truncations.get(a, False)]
        for agent in agents_to_remove:
            self.agents.remove(agent)
            self.terminations.pop(agent, None)
            self.truncations.pop(agent, None)
            self.rewards.pop(agent, None)
            self.current_actions.pop(agent, None)
            self.infos.pop(agent, None)

        # Generate observation and rewards
        observations = {agent: self.observe(agent) for agent in self.agents}
        rewards = {agent: self.rewards.get(agent, 0) for agent in self.agents}

        # Reset rewards for next step
        for agent in self.agents:
            self.rewards[agent] = 0

        return observations, rewards, self.terminations, self.truncations, self.infos


    def last(self, observe: bool = True):
        """ Returns the final obsrvations and the accumulated rewards """
        if not self.agents:
            return {}, {},{},{} # returns empy dictionaries for consistencies

        # Collect observations of all agents
        observations = {agent: self.observe(agent) for agent in self.agents} if observe else {}

        #Get final accumulated rewards
        rewards = {agent: self._cumulative_rewards.get(agent, 0) for agent in self.agents}

        # Reset rewards
        for agent in self.agents:
            self._cumulative_rewards[agent] = 0

        print(f"[DEBUG] Step() -> Next agent: {self.agent_selection}")
        print(f"[DEBUG] Last() -> Agent Selection: {self.agent_selection}, Active Agents: {self.agents}")

        return observation, rewards, self.terminations, self.truncations, self.infos
    ########### OBSERVATION  ##########
    def observe(self, agent):
        """Structured observation for customers"""

        # Initialiize empty observations
        observations = {}

        # Check if agent exists in Observation space
        for agent in self.agents:
            if agent not in self.observation_spaces:
                print(f"Agent {agent} is not found in Observation Space! Empty observation will be returned")
                observations[agent] = np.zeros(self.max_obs_size, dtype=np.float32)
                continue  # skip further processing for this agent

            obs_dict = self.get_agent_observation(agent)

        if not obs_dict or not all(isinstance(v, (list, np.ndarray)) for v in obs_dict.values()):
            print(f"[DEBUG] Invalid obs_dict for {agent}: {obs_dict}")
            return np.zeros(self.max_obs_size, dtype=np.float32)  # ✅ Ensure valid format

        # convert dict to a numpy array
        obs_array = np.concatenate([np.array(v).flatten() for v in obs_dict.values()])

        #  Flattten and Pad Observation if necessary to fit the space
        flat_obs = spaces.utils.flatten(self.observation_spaces[agent], obs_array)
        padded_obs = np.zeros(self.max_obs_size, dtype=np.float32)
        padded_obs[:min(flat_obs.shape[0], self.max_obs_size)] = flat_obs[:self.max_obs_size]

        #Clip Observation to ensure valid space limits
        clipped_obs = np.clip(
            padded_obs,
            self.observation_spaces[agent].low.min(),
            self.observation_spaces[agent].high.max()
        )

        # Ensure agent exists in `infos`
        if agent not in self.infos:
            print(f"Agent {agent} is missing from env.infos! Current infos: {self.infos}")

        observations[agent] = clipped_obs

        return observations

    def state(self):
        """Returns a Global state of the environment."""
        return {
            "market_trend": self.market_trend,
            "total_transactions": self.total_transactions,
            "vendor_revenues": self.vendor_revenue,
            "customer_budgets": self.customer_budgets,
        }
    ########### GET OBSERVATIONS #############
    def get_agent_observation(self, agent):
        """"Observation retrieval for multiple agents"""

        observations = {}
        for agent in self.agents:
            if agent not in self.agents:
                observations[agent] = {}
                continue

        # CUSTOMER OBSERVATION
        if agent.startswith("customer"):
            observations[agent] = {
                "vendors": np.array([
                        1 if self.vendors_products[v] else 0
                        for v in self.vendors_products]
                ),
                "prices": np.clip(np.array([(
                            self.round_price(np.mean([p.price for p in self.vendors_products[v]]))
                            if self.vendors_products[v]
                            else 0)
                        for v in self.vendors_products],
                    dtype=np.float32) / 500.0, 0, 1),
                "stock": np.clip(np.array([(
                            sum(p.stock for p in self.vendors_products[v])
                            if self.vendors_products[v]
                            else 0)
                        for v in self.vendors_products],
                    dtype=np.int32) / 200.0, 0, 1),
                "market_trend": self.market_trend,
                "budget": np.array([self.round_price(self.customer_budgets.get(agent, 0))], dtype=np.float32),
            }

        # VENDOR OBSERVATION
        elif agent.startswith("vendor"):
            observations[agent] = {
                "competitor_prices": np.array([(
                            self.round_price(np.mean([p.price for p in self.vendors_products[v]]))
                            if self.vendors_products[v]
                            else 0)
                        for v in self.vendors_products
                        if v != agent],
                    dtype=np.float32,
                ),
                "own_stock": np.clip(np.array(
                    [sum(p.stock for p in self.vendors_products[agent])], dtype=np.float32) / 200.0, 0, 1
                ),
                "market_demand": self.market_trend,
                "sales_history": np.clip(np.array((
                        self.market_price_history[agent][-5:]
                        if len(self.market_price_history[agent]) >= 5
                        else self.market_price_history[agent] + [0] * (5 - len(self.market_price_history[agent]))),
                    dtype=np.float32) / 500.0, 0, 1
                ),
                "status": 1 if self.vendors_status[agent] == "active" else 0,
            }
        return observations.get(agent, {})


    def _remove_terminated_agents(self):
        """Removes terminated agents"""
        terminated_agents = {agent for agent in self.agents if self.terminations.get(agent, False)}

        # Remove agents effieciently
        self.agents = [agent for agent in self.agents if agent not in terminated_agents]
        self.possible_agents = self.agents[:]
    #########  REWARD FUNCTION ############
    def select_vendor(self, customer):
        """Selects multiple vendors for customers in parallel"""
        vendors_list = [
            v for v in self.vendors_products if self.vendors_products[v]]
        # Randomly assigns a vendor to a customer
        return {
            customer: np.random.choice(vendors_list) if vendors_list else None
        for customer in customers
        }

    def select_product(self, vendor):
        """Selects single product for customers in parallel"""

        return {
            vendor: np.random.choice(self.vendors_products[vendor])
            if vendor in self.vendors_products and self.vendors_products[vendor]
            else None
            for vendor in vendors
        }


    def check_done(self):
        """
        Checks if agents are done and updates self.terminations accordingly.
        A customer is done if their budget is ≤ 0.
        A vendor is done if they have no products left.
        The environment is done if all customers or all vendors are done.
        """

        done_customers = {customer for customer in self.customer_budgets
            if self.customer_budgets[customer] <= 0
        }
        done_vendors = {vendor for vendor in self.vendors_products
            if not self.vendors_products[vendor]
        }
        print(f"[DEBUG] Done Customers: {done_customers} | Done Vendors: {done_vendors}")

        all_customers_done = len(done_customers) == len(self.customer_budgets)
        all_vendors_done = len(done_vendors) == len(self.vendors_products)

        #Track removed agents
        if all_customers_done or all_vendors_done:
            print(f"[DEBUG] All agents are done. Terminating environment.")
            self.done_agents.update(done_customers | done_vendors)
            self.agents.clear()
            self.possible_agents.clear()
            return True

        terminated_agents = done_customers | done_vendors

        #Batch update for terminated agents
        for agent in terminated_agents:
            self.terminations[agent] = True
            self.truncations[agent] = True
            print(f"[DEBUG] check_done() -> Marked {agent} as terminated.")
        # Delete agents completely
        for agent in terminated_agents:
            print(f"[DEBUG] check_done() -> Removing {agent} from terminations.")
            self.truncations.pop(agent, None)
            self.rewards.pop(agent, None)
            self.current_actions.pop(agent, None)
            self.infos.pop(agent, None)

        if not self.agents:
            print(f"[DEBUG] All agents are done. Clearing environment.")
            self.agents.clear()
            self.possible_agents.clear()
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

        # Calculate price change per trends
        price_trends = [
            (history[-1] - history[0]) / history[0] if len(history) ==3 and history[0] != 0 else 0
            for history in last_prices
        ]

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


        # Count active or inactive vendors
        self.active_vendors = sum(1 for v in self.vendors_status.values() if v == "active")
        self.inactive_vendors = self.vendors - self.active_vendors

        self.total_transactions = sum(len(products) for products in self.vendors_products.values())
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
        """Updates stock for vendors"""
        if product in self.vendors_products[vendor]: # Check if product exist with the vendor
            product.update_stock(quantity)
        else:
            print(f"{vendor} does not have {product}")

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