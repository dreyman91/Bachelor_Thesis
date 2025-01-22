class RobotFactoryEnv:
    def __init__(self):
        self.agents = ["robot_1", "robot_2"]
        self.current_agent = 0
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
        agent = self.agents[self]