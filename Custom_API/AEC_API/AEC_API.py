class SimpleAECEnv:
    def __init__(self): #Creates two agents and initializes the environment
        self.agents = ["agent_1", "agent_2"]
        self.current_agent_index = 0
        self.terminations = {agent: False for agent in self.agents}

    def reset(self):
        """"Reset the Environment"""
        print("Resetting the environment....")
        self.current_agent_index = 0
        self.terminations = {agent: False for agent in self.agents}

    def observe(self, agent):
        """Returns an observation for the current agent """
        return f"Observation for {agent}"

    def action(self, action):
        """Steps forward in the environment"""
        agent = self.agents[self.current_agent_index]
        print(f"{agent} takes action: {action}")

        #Check if agent should be terminated
        if action == "quit":
            self.terminations[agent] = True
            print(f"{agent} is done!")

        #Move to next agent
        self.current_agent_index = (self.current_agent_index + 1) % len(self.agents)

    def agent_iter(self ):
        """Iterator to cycle through agents until all are done"""
        while not all(self.terminations.values()):
            yield self.agents[self.current_agent_index] #Return next agent


#Testing the API
env = SimpleAECEnv()
env.reset()

for agent in env.agent_iter():
    print(f"Current agent: {agent}")
    obs = env.observe(agent)
    print(obs)
    action = input("Enter an action(move/quit): ")
    env.step(action)