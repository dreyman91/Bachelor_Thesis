import random
import numpy as np
from pettingzoo import AECEnv
from pettingzoo.utils import wrappers
from gymnasium import spaces

class RobotFactoryEnv(AECEnv):
    metadata = {'render.modes': ['human', 'rgb_array'], "name": "robot_factory_v1"}

    def __init__(self):
        super().__init__()
        self.agents = [f"robots_{i}" for i in range(1,6)]
        self.possible_agents = self.agents[:]
        self.agent_index = 0

        self.terminations = {agent: False for agent in self.agents}
        self.task_status = {agent: "Idle" for agent in self.agents}
        self.energy_levels = {agent: 100 for agent in self.agents}
        self.task_queue = {agent: [random.choice(["Pick", "Move", "Assemble Part",
                                                "Inspect Quality", "Charge"]) for _ in range(3)]
                           for agent in self.agents}
        self.obstacles = {f"obstacle_{random.randint(1, 10)}" for _ in range(3)}


        # Define action and observation spaces
        self.action_spaces = {agent: spaces.Discrete(6) for agent in self.agents}
        self.observation_spaces = {
            agent: spaces.Dict({
                "energy": spaces.Box(low=0, high=100, shape=(1, ), dtype=np.float32),
                "tasks left": spaces.Discrete(5),
                "obstacles nearby": spaces.Discrete(10),
            }) for agent in self.agents
        }

    def reset(self,seed=None, options=None):
        """Resets the environment"""
        self.agents = self.possible_agents[:]
        self.agent_index = 0
        self.terminations = {agent: False for agent in self.agents}
        self.task_status = { agent: "Idle" for agent in self.agents}
        self.energy_levels = {agent: 100 for agent in self.agents}
        self._agent_selector = iter(self.agents)
        self.agent_selection = next(self._agent_selector)
        self._cumulative_reward = {agent: 0 for agent in self.agents}

    # Define Observation spaces
    def observe(self, agent):
        """Returns observation of agent_i"""
        if agent not in self.agents:
            return None
        return{
            "energy": np.array([self.energy_levels[agent]], dtype=np.float32),
            "tasks_left": len(self.task_queue[agent]),
            "obstacles_nearby": len(self.obstacles)
        }

    def step(self, action):
        """Executes an action for the current agent"""
        agent = self.agent_selection

        actions_map = {
            0: "Pick",
            1: "Move",
            2: "Assemble",
            3: "Inspect",
            4: "Charge",
            5: "Shutdown"}

        selected_action = actions_map[action]
        print(f"➡️ {agent} executes: {selected_action}")

        if action == "Pick":
            if self.task_queue[agent]:
                self.task_status[agent] = self.task_status[agent].pop(0)
                print(f"{agent} picked task: {self.task_status[agent]}")

        elif action == "Move":
            if random.random() < 0.2:
                print(f"{agent} encountered an obstacle and lost 10 energy")
                self.energy_levels[agent] -= 10
            else:
                self.task_status[agent] = "Moving"

        elif action == "Charge":
            if self.energy_levels[agent] == 100:
                print(f"{agent} is fully charged")
            else:
                print(f"{agent} is still charging")

        elif selected_action == "Assemble":
            if random.random() < 0.8:
                self.task_status[agent] = "Assembly Completed"
                print(f"🛠️ SUCCESS: {agent} assembled a part!")
            else:
                self.task_status[agent] = "Assembly Failed"
                print(f"⚠️ FAILED: {agent} failed assembly, needs retry!")

        elif selected_action == "Inspect":
            if random.random() < 0.9:
                self.task_status[agent] = "Inspection Passed"
                print(f"✅ SUCCESS: {agent} passed inspection!")
            else:
                self.task_status[agent] = "Inspection Failed"
                print(f"❌ FAILED: {agent} failed inspection, needs rework.")



        elif action == "Shutdown":
            self.terminations[agent] = True
            print(f"{agent} is shutting down")

        self.energy_levels[agent] -= random.randint(5, 15)
        if self.energy_levels[agent] <= 0:
            self.terminations[agent] = True
            print(f"{agent} ran out of energy")

        self.agent_selection = next(self._agent_selector, None)
        if self.agent_selection is None:
            self.agent_selection = iter(self.agents)



    # def agent_iter(self):
    #     """Loops through the available agents"""
    #     while not all(self.terminations.values()):
    #         agents_available = [agent for agent in self.agents if not self.terminations[agent]]
    #         random_shuffle(agents_available)
    #         for agent in agents_available:
    #             yield agent


    def render(self):
        """Displays the current state of the environment"""
        print(f"robot factory state:")
        for agent in self.agents:
            print(f"{agent}: | Status: {self.task_status[agent]}"
                  f"| Energy: {self.energy_levels[agent]}"
                  f"| Task left: {self.task_status[agent]}"
                  f"| Obstacles: {len(self.obstacles)}")
    def close(self):
        """Cleans up the environment."""
        pass






# env = RobotFactoryEnv()
# env.reset()
#
# for agent in env.agent_iter():
#     print(f"\n Current Robot: {agent}")
#     obs = env.observe(agent)
#     print(obs)
#     print("Enter action: (1) Pick, (2) Move, (3) Assemble, (4) Inspect, (5) Charge, (6) Shutdown")
#     try:
#         action_num = int(input("Select an action (1-6): "))
#         env.step(agent, action_num)
#     except ValueError:
#         print("❌ Invalid input! Please enter a number between 1 and 6.")



