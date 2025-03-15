
r"""

Market_AEC_Env: A multi-agent reinforcement learning environment \n
for simulating a market with customers, vendors, and administrators.

Author: Oluwadamilare Adegun
Date: 10.01.2025.
"""

from pettingzoo import ParallelEnv
import numpy as np
from gymnasium import spaces
from decimal import Decimal, ROUND_HALF_UP


########## PRODUCT CLASS ##########

def _format_agent_name( role: str, index: int):
    if role not in ["customer", "vendor"]:
        raise ValueError(f"Role {role} is not supported. Must be customer or vendor.")
    return f"{role}_{index}"

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


    metadata = {'render.modes': ['human', 'rgb_array'], "name": "market_v0"}

    ########### INITIALIZATION ##########
    def __init__(self, vendors=3, customers=3, render_mode="human", max_steps=100):
        """"Initialization"""
        super().__init__()
        self.vendors = vendors
        self.customers = customers
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.step_count = 0

        # Agent IDs
        self.possible_agents =[
            _format_agent_name("customer", i) for i in range(customers)
        ] + [_format_agent_name("vendor", i) for i in range(vendors)]
        self.agents = self.possible_agents

        self.action_spaces = {
            agent: spaces.Discrete(2) if "customer" in agent else spaces.Discrete(4)
            for agent in self.possible_agents
        }
        self.observation_spaces = {
            agent: spaces.Box(low=0, high=1, shape=(5,), dtype=np.float32)
            for agent in self.possible_agents
        }

    def _initialize_agents(self):
        """ Initialize agent specific data structures """
        pass

    def reset(self, seed=None, options=None):
        """ Resets the environment for new episode"""
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        else:
            self.np_random = np.random.default_rng()
        self.step_count = 0
        self.agents = self.possible_agents[:]

        self.customer_budgets = {
            _format_agent_name("customer", i): np.random.randint(50, 2000)
            for i in range(self.customers)
        }
        ####### VENDOR ATTRIBUTES ########
        self.vendors_products = {
            _format_agent_name("vendor", i): [
                Product(
                    product_id=1,
                    price=self.np_random.uniform(500, 1000),
                    name="Iphone",
                    category="Electronics",
                    stock=self.np_random.integers(20, 50),
                ),
                Product(
                    product_id=2,
                    price=self.np_random.uniform(10, 30),
                    name="Apple",
                    category="Fruit",
                    stock=self.np_random.integers(20, 100),
                ),
                Product(
                    product_id=3,
                    price=self.np_random.uniform(5, 15),
                    name="Coca-Cola",
                    category="Beverages",
                    stock=self.np_random.integers(80, 100)
                ),
            ]
            for i in range(self.vendors)
        }

        self.vendor_revenue = {agent: 0 for agent in self.possible_agents if agent.startswith("vendor")}
        self.vendors_status = {agent: "active" for agent in self.possible_agents if agent.startswith("vendor")}
        self.market_prices = {agent: 0 for agent in self.possible_agents if agent.startswith("vendor")}

        self.terminations = {agent: False for agent in self.possible_agents}
        self.truncations = {agent: False for agent in self.possible_agents}
        self._cumulative_rewards = {agent: 0 for agent in self.possible_agents}
        self.rewards = {agent: 0 for agent in self.possible_agents}
        self.current_actions = {agent: None for agent in self.possible_agents}
        self.infos = {agent: {} for agent in self.possible_agents}

        observations = {agent: self.observe(agent) for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        return observations, infos


    def step(self, actions):
        """Agent takes action based on observation"""
        if not isinstance(actions, dict):
            raise ValueError("Actions must be a dictionary.")

        for agent  in self.agents:
            self.rewards[agent] = 0

        for agent, action in actions.items():
            if agent not in self.agents:
                continue

            if agent.startswith("customer"):
                if action == 0:
                    self._customer_buy(agent)
            elif agent.startswith("vendor"):
                if action == 0:
                    self._vendor_adjust_price(agent)

        self.step_count += 1
        self._check_termination()

        observations = {agent:self.observe(agent) for agent in self.agents}
        rewards = self.rewards.copy()
        terminations = {agent: agent not in self.agents for agent in self.possible_agents}
        truncations = {agent: self.step_count >= self.max_steps for agent in self.possible_agents}
        infos = {agent: {} for agent in self.agents}

        return observations, rewards, terminations, truncations, infos

    def _check_termination(self):
        """ checks for episode termination """
        if all(self.customer_budgets[agent] <= 0 for agent in self.agents if agent.startswith("customer")):
            for agent in self.agents:
                self.terminations[agent] = True
            self.agents = []

        if all(not self.vendors_products[agent] for agent in self.agents if agent.startswith("vendor")):
            for agent in self.agents:
                self.terminations[agent] = True
            self.agents = []

        if self.step_count >= self.max_steps:
            for agent in self.agents:
                self.truncations[agent] = True
            self.agents = []

    def _customer_buy(self, customer):
        vendor = self.np_random.choice([agent for agent in self.agents if agent.startswith("vendor")])

        if not vendor  or not self.vendors_products.get(vendor):
            self.rewards[customer] -= 1
            return
        product = self.np_random.choice(self.vendors_products[vendor])

        if self.customer_budgets[customer] >= product.price and product.stock > 0:
            self.customer_budgets[customer] -= product.price
            product.update_stock(-1)
            self.rewards[customer] += 5
            self.rewards[vendor] += 5
        else:
            self.rewards[customer] -= 2

    def _vendor_adjust_price(self, vendor):
        if not self.vendors_products.get(vendor):
            return

        product = self.np_random.choice(self.vendors_products[vendor])
        change = self.np_random.uniform(-0.5, 0.5)
        product.set_price(max(1, product.price + change))

    def observe(self, agent):
        if agent.startswith("customer"):
            avg_price = np.mean([
                p.price
                for v in self.vendors_products
                for p in self.vendors_products[v]
            ])
            observation = np.array(
                [avg_price / 10.0, self.customer_budgets[agent] / 200.0, 0.0, 0.0, 0.0],
                dtype=np.float32,
            )
        elif agent.startswith("vendor"):
            if not self.vendors_products.get(agent):
                observation = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32,)
            else:
                own_stock = self.vendors_products[agent][0].stock
                competitor_prices = [
                    p.price
                    for v in self.vendors_products
                    for p in self.vendors_products[v]
                    if v != agent
                ]
                avg_competitor_price = (
                    np.mean(competitor_prices) if competitor_prices else 0
                )
                observation = np.array(
                    [own_stock / 100.0, avg_competitor_price / 10.0, 0.0, 0.0, 0.0], dtype=np.float32,
                )
        else:
            observation = np.zeros(5, dtype=np.float32)
        return observation

    #########  RENDER ############
    def render(self, mode="human"):
        if mode == "human":
            print(f"Step: {self.step_count}")
            for agent in self.agents:
                if agent.startswith("customer"):
                    print(f"{agent}: Budget = {self.customer_budgets[agent]:.2f}")
                elif agent.startswith("vendor"):
                    if not self.vendors_products.get(agent):
                        print(f"{agent}: Out of Stock, Reward = {self.rewards[agent]:.2f}")
                    else:
                        product = self.vendors_products[agent][0]
                        print(f"{agent}: {product.name} Price = {product.price:.2f}")
        print("-" * 20)
    def close(self):
        print("Market environment closed.")