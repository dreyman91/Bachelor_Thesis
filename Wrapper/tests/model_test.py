from Wrapper.src.wrapper_api.wrappers.base_wrapper import ActiveCommunication

agents = ["agent_0", "agent_1"]
comm = ActiveCommunication(agent_ids=agents)

print("Matrix dtype:", comm.matrix.dtype)
print("Initial Matrix:", comm.get_state())

comm.update("agent_0", "agent_1", False)
print("Can agent_0 talk to agent_1?", comm.can_communicate("agent_0", "agent_1"))

print("Blocked for agent_0:", comm.get_blocked_agents("agent_0"))  # Expect: ['agent_1']


comm.reset()
print("After reset:\n", comm.get_state())