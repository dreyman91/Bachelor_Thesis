import numpy as np
from failure_api.communication_models.markov_model import BaseMarkovModel
from failure_api.communication_models import ActiveCommunication

# --- Test setup ---
agent_ids = ["a1", "a2"]

# Transition matrix that always transitions to 1 (connected)
T = np.array([
    [0.0, 1.0],  # if disconnected, always connect
    [0.0, 1.0],  # if connected, stay connected
])
transitions = {("a1", "a2"): T, ("a2", "a1"): T}

# Initialize model with deterministic behavior
model = BaseMarkovModel(agent_ids, transition_probabilities=transitions)
model.rng = np.random.RandomState(seed=42)  # deterministic randomness

# Initialize communication matrix
comms = ActiveCommunication(agent_ids)

# --- Assert initial state is 1 (connected) for all pairs ---
for s in agent_ids:
    for r in agent_ids:
        if s != r:
            assert model.state[(s, r)] == 1, f"Initial state of ({s}->{r}) should be 1"

# --- Run connectivity update ---
model.update_connectivity(comms)

# --- Assert communication matrix reflects always-connected logic ---
matrix = comms.get_boolean_matrix()
for s in agent_ids:
    for r in agent_ids:
        if s != r:
            assert matrix[agent_ids.index(s), agent_ids.index(r)] == True, f"{s}->{r} should be connected"
        else:
            assert matrix[agent_ids.index(s), agent_ids.index(r)] == False, f"{s}->{r} should not be self-connected"

print("tests passed.")
