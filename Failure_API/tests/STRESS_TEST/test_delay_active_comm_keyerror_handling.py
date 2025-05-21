#=====================test_comm_model_agent_key_failures==================#

import pytest
import numpy as np
from Failure_API.src.failure_api.communication_models.active_communication import ActiveCommunication
from Failure_API.src.failure_api.communication_models.delay_based_model import DelayBasedModel
from Failure_API.src.failure_api.wrappers import CommunicationWrapper
from collections import deque
from mpe2 import simple_spread_v3


def test_missing_shared_obs_causes_typeerror():
    # Step 1: Create environment
    base_env = simple_spread_v3.env(N=3, max_cycles=25)
    failure_model = DelayBasedModel(agent_ids=["agent_0", "agent_1", "agent_2"], min_delay=1, max_delay=2)

    # Step 2: Wrap with CommunicationWrapper (which wraps SharedObsWrapper inside)
    env = CommunicationWrapper(base_env, failure_models=[failure_model])

    # Step 3: Reset the env (observations haven't been populated yet)
    env.reset()

    # Step 4: Try calling .observe() manually before stepping
    # This triggers the bug (no shared observations computed yet)
    with pytest.raises(TypeError) as e:
        _ = env.observe("agent_0")

    assert "Expected dictionary from SharedObsWrapper" in str(e.value)


def test_shared_obs_available_after_step():
    env = CommunicationWrapper(
        simple_spread_v3.env(N=3, max_cycles=25),
        failure_models=[DelayBasedModel(["agent_0", "agent_1", "agent_2"], 1, 2)]
    )
    env.reset()

    # Trigger step (which should initialize shared observations)
    agent = env.agent_selection
    action = env.action_space(agent).sample()
    env.step(action)

    print("Type of env:", type(env))
    print("Type of env.env:", type(env.env))
    print("Type of env.env.env (base):", type(env.env.env))

    obs = env.observe(agent)
    assert isinstance(obs, dict), f"Expected dict after step, got {type(obs)}"

# def test_agent_sync_failure_between_env_and_comms():
#     # Step 1: Create base environment and step to get agent names
#     base_env = simple_spread_v3.env(N=3, local_ratio=0.5, max_cycles=25)
#     base_env.reset()
#     agents_from_env = list(base_env.agents)
#
#     # Step 2: Create mismatched communication model (agent_0 missing)
#     bad_agent_ids = ["agent_1", "agent_2"]  # Missing 'agent_0' intentionally
#     delay_model = DelayBasedModel(agent_ids=bad_agent_ids, min_delay=1, max_delay=2)
#
#     # Step 3: Create ActiveCommunication with the same bad agent list
#     comms_matrix = ActiveCommunication(agent_ids=bad_agent_ids)
#
#     # Step 4: Wrap with CommunicationWrapper (manually injected)
#     env = CommunicationWrapper(base_env, failure_models=[delay_model], comms_matrix=comms_matrix)
#
#     # Step 5: Reset to get environment into ready state
#     env.reset()
#
#     # Step 6: Call .step() and expect failure due to agent_0 being active in env.agents but missing in comms_matrix
#     actions = {agent: env.action_space(agent).sample() for agent in env.agents}
#     with pytest.raises(KeyError) as excinfo:
#         env.step(actions)
#
#     # Step 7: Assert meaningful KeyError message
#     assert "agent_0" in str(excinfo.value), f"Expected KeyError due to missing agent_0, got: {excinfo.value}"







# def test_comm_model_agent_key_failures():
#     # Valid agent IDs
#     agent_ids = ["agent_0", "agent_1", "agent_2"]
#
#     # Initialize communication matrix
#     comms = ActiveCommunication(agent_ids=agent_ids)
#
#     # Test ActiveCommunication KeyError
#     with pytest.raises(KeyError) as e1:
#         comms.update("agent_0", "ghost_agent", 1.0)
#     assert "ghost_agent" in str(e1.value)
#
#     # Initialize DelayBasedModel with proper IDs
#     delay_model = DelayBasedModel(agent_ids=agent_ids, min_delay=1, max_delay=3)
#
#     # Create a fake comms matrix with the correct agents
#     comms_matrix = ActiveCommunication(agent_ids=agent_ids)
#
#     # Inject a bad message into the delay queue
#     delay_model.message_queues[("agent_0", "ghost_agent")] = deque([
#         (0, True)
#     ])
#
#     # Expect failure due to 'ghost_agent' not being in comms_matrix
#     with pytest.raises(KeyError) as e2:
#         delay_model.process_existing_messages(comms_matrix)
#     assert "ghost_agent" in str(e2.value)
#
#     # Additional edge case: unknown sender
#     delay_model.message_queues[("ghost_sender", "agent_1")] = deque([
#         (0, True)
#     ])
#     with pytest.raises(KeyError) as e3:
#         delay_model.process_existing_messages(comms_matrix)
#     assert "ghost_sender" in str(e3.value)


# def test_delay_model_rejects_unknown_agents():
#     # Known valid agents
#     agent_ids = ["agent_0", "agent_1"]
#     comms_matrix = ActiveCommunication(agent_ids=agent_ids)
#     delay_model = DelayBasedModel(agent_ids=agent_ids, min_delay=1, max_delay=2)
#
#     # Inject invalid sender into the queue
#     delay_model.message_queues[("ghost_sender", "agent_0")] = deque([(0, True)])
#
#     # Inject invalid receiver into the queue
#     delay_model.message_queues[("agent_0", "ghost_receiver")] = deque([(0, True)])
#
#     # Should fail clearly when processing invalid agents
#     with pytest.raises(KeyError) as e1:
#         delay_model.process_existing_messages(comms_matrix)
#     assert "ghost_sender" in str(e1.value) or "ghost_receiver" in str(e1.value)

# #=====================test_comm_model_invalid_receiver==================#
# def test_comm_model_invalid_receiver():
#     from collections import deque
#     agent_ids = ["agent_0", "agent_1", "agent_2"]
#     delay_model = DelayBasedModel(agent_ids=agent_ids, min_delay=1, max_delay=3)
#     comms_matrix = ActiveCommunication(agent_ids=agent_ids)
#
#     # Inject only bad receiver
#     delay_model.message_queues[("agent_0", "ghost_agent")] = deque([(0, True)])
#
#     with pytest.raises(KeyError) as e:
#         delay_model.process_existing_messages(comms_matrix)
#     assert "ghost_agent" in str(e.value)


# #=====================test_comm_model_invalid_sender==================#
# def test_comm_model_invalid_sender():
#     from collections import deque
#     agent_ids = ["agent_0", "agent_1", "agent_2"]
#     delay_model = DelayBasedModel(agent_ids=agent_ids, min_delay=1, max_delay=3)
#     comms_matrix = ActiveCommunication(agent_ids=agent_ids)
#
#     # Inject only bad sender
#     delay_model.message_queues[("ghost_sender", "agent_1")] = deque([(0, True)])
#
#     with pytest.raises(KeyError) as e:
#         delay_model.process_existing_messages(comms_matrix)
#     assert "ghost_sender" in str(e.value)