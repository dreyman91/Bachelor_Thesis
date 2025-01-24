import queue
import random

class AEC_Enhanced_Robot:
    def __init__(self):
        """Initialize AEC Enhanced Robot environment with 5 robots."""
        self.agents = [f"robot{i}" for i in range(1,6)]
        self.current_agent_index = 0
        self.terminations = {agent: False for agent in self.agents}
        self.task_status = {agent: "Idle" for agent in self.agents}
        self.energy_levels = {agent: 100 for agent in self.agents}
        self.task_queue = {}
        self.obstacles= set()

        self.generate_random_tasks()
        self.generate_obstacles()

    def generate_random_tasks(self):
        """Generate random tasks for each robot."""
        possible_tasks = ["Pick Item", "Move Item", "Assemble Part", "Inspect Quality", "Charge"]
        for agent in self.agents:
            self.task_queue[agent] = [random.choice(possible_tasks) for _ in range(random.randint(1,5))]

    def generate_obstacles(self):
        """Generate random obstacles for each robot."""
        for _ in range(3):
            self.obstacles.add(f"obstacle{random.randint(1,10)}")

    def reset(self  ):
        """Reset the environment to its initial state."""
        print("\n🔄 Resetting environment")
        self.current_agent_index = 0
        self.terminations = {agent: False for agent in self.agents}
        self.task_status = {agent: "Idle" for agent in self.agents}
        self.energy_levels = {agent: 100 for agent in self.agents}
        self.generate_random_tasks()
        self.generate_obstacles()

    def observe(self, agent):
        """Returns the Observation of the agent."""
        return (f"{agent} | Status: {self.task_status[agent]} | Energy: {self.energy_levels[agent]} | "
                f"Tasks Left: {len(self.task_queue[agent])} | Obstacles: {self.obstacles}")

    def step(self, agent, action_num):
        """Processes an agent's action using numerical input"""
        ACTION_MAP = {
            1: "Pick Item",
            2: "Move",
            3: "Assemble",
            4: "Inspect",
            5: "Charge",
            6: "Shutdown"
        }

        if action_num not in ACTION_MAP:
            print("❌ Invalid action! Please enter a number between 1 and 6.")
            return

        action = ACTION_MAP[action_num]
        print(f"➡️ {agent} executes: {action}")



        if action == "Pick Item":
            if self.task_queue[agent]:
                task = self.task_queue[agent].pop(0) # Take next task
                self.task_status[agent] = task
                print(f"✅ {agent} picked task: {task}")
            else:
                print("⚠️no task available!")


        elif action == "Move":
            if random.random() < 0.2: # 20% chance of hitting an obstacle
                print(f"⚠️ {agent} encountered an obstacle and lost 10 energy!")
                self.energy_levels[agent] -= 10
            else:
                self.task_status[agent] = "Moving"
                print(f"🏂 {agent} is moving...")

        elif action == "Assemble Part":
            if random.random() < 0.8:
                self.task_status[agent] = "Assembly completed"
                print (f"{agent} successfully assembled a part")

        elif action == "Inspect Quality":
            if random.random() < 0.7:
                self.task_status[agent] = "Inspection failed"
                print(f"❌ {agent} failed inspection. Needs check!")
            else:
                self.task_status[agent] = "Inspection Passed"
                print(f"✅ {agent} passed  inspection. Ready for task!")

        elif action == "Charge":
            self.energy_levels[agent] = 100
            print(f"🔋 {agent} energy level is full!")

        elif action == "Shutdown":
            self.terminations[agent] = True
            print(f"😴 {agent} is shutting down!")

        # Reduce energy on every step
        self.energy_levels[agent] -= random.randint(5,15)
        if self.energy_levels[agent] <= 0:
            self.terminations[agent] = True
            self.task_status[agent] = "Out of Energy"
            print(f"‼️ {agent} ran out of energy and shut down!")

        if agent in self.agents:
            self.agents.remove(agent)

        # Move to next agent
        self.current_agent_index = (self.current_agent_index + 1) % len(self.agents)

    def agent_iter(self):
        """Agent turn-based loop execution"""
        while not all(self.terminations.values()):
            available_agents = [agent for agent in self.agents if not self.terminations[agent]]
            random.shuffle(available_agents)
            for agent in available_agents:
                yield agent


env = AEC_Enhanced_Robot()
env.reset()

for agent in env.agent_iter():
    print(f"\n🤖 Current Robot:", {agent})
    obs = env.observe(agent)
    print(obs)
    print("Enter action: (1) Pick, (2) Move, (3) Assemble, (4) Inspect, (5) Charge, (6) Shutdown")
    try:
        action_num = int(input("Select an action (1-6): "))
        env.step(agent, action_num)
    except ValueError:
        print("❌ Invalid input! Please enter a number between 1 and 6.")
