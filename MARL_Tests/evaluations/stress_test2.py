import unittest
from mpe2 import simple_spread_v3
from pettingzoo.utils import parallel_to_aec, aec_to_parallel
from failure_api.wrappers import CommunicationWrapper
from failure_api.communication_models import ProbabilisticModel


class TestAgentSync(unittest.TestCase):

    def test_agents_sync_after_step(self):

        # Create base environment
        raw_env = simple_spread_v3.env(N=3, max_cycles=5)

        # Wrap BEFORE converting to AEC (so the wrapper is in AEC form)
        comm_model = ProbabilisticModel(agent_ids=raw_env.possible_agents, failure_prob=0.0)
        wrapped_env = CommunicationWrapper(raw_env, failure_models=[comm_model])

        # Now convert to AEC safely (CommunicationWrapper should subclass AECEnv)
        par_env = aec_to_parallel(wrapped_env)

        # Safe reset (check if env.reset returns a tuple)
        obs, infos = par_env.reset(seed=42)

        for step in range(20):
            actions = {
                agent: par_env.action_space(agent).sample()
                for agent in par_env.agents
            }
            obs, rewards, terminations, truncations, infos = par_env.step(actions)

            # Assert .agents is synced between wrapped and base
            self.assertEqual(wrapped_env.agents, par_env.agents,
                             msg=f"Desync at step {step}: {wrapped_env.agents} vs {par_env.agents}")
            if not par_env.agents:
                break


if __name__ == "__main__":
    unittest.main()
