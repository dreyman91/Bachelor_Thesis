class RobotFactoryEnv:
    def __init__(self):
        self.agents = ["robot_1", "robot_2"]
        self.current_agent_index = 0
        self.terminations = {agent: False for agent in self.agents}
        self.task_status = {"robot_1": "Idle", "robot_2": "Idle"} #Tracking their task

    def reset(self  ):
        print("Factory Resetting...")
        self.current_agent = 0
        self.terminations = { agent: False for agent in self.agents }
        self.task_status = {"robot_1": "Idle", "robot_2": "Idle"}

    def observe(self, agent):
        return f"{agent} is currently {self.task_status[agent]}"

    def step(self, action ):
        agent = self.agents[self.current_agent_index]
        print(f"{agent} executes: {action}")

        if action == 'pick':
            self.task_status[agent] = "Picked item"
        elif action == 'move':
            self.task_status[agent] = "Moved item"
        elif action == 'shutdown':
            self.terminations[agent] = True
            print(f"Agent is shutting down!")

        self.current_agent  = (self.current_agent + 1) % len(self.agents)

    def agent_iter(self ):
        while not all(self.terminations.values()):
            yield self.agents[self.current_agent_index]

env = RobotFactoryEnv()
env.reset()

for agent in env.agent_iter():
    print(f"Current robot: {agent}")
    obs = env.observe(agent)
    print(obs)
    action = input("Enter action (pick, move, shutdown): ")
    env.step(action)