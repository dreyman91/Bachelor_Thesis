from pettingzoo.utils.wrappers import BaseWrapper
from pettingzoo.utils.env import AgentID, AECEnv, ActionType, ObsType
import numpy as np
from gymnasium import spaces
from typing import Optional, List, Any
from Failure_API.src.failure_api.communication_models.active_communication import ActiveCommunication
from Failure_API.src.failure_api.communication_models.base_communication_model import CommunicationModels
from Failure_API.src.failure_api.communication_models.probabilistic_model import ProbabilisticModel
from Failure_API.src.failure_api.wrappers.sharedobs_wrapper import SharedObsWrapper



class CommunicationWrapper(BaseWrapper):
    """
    A wrapper for PettingZoo environments that simulates communication failures between agents.

    This wrapper applies various communication failure models to control which agents can
    communicate with each other, and how their observations and actions are affected by
    communication constraints.
    """
    def __init__(self,
                 env: AECEnv[AgentID, ActionType, ObsType],
                 failure_models: List[CommunicationModels] = None,
                 agent_ids: Optional[List[str]] = None
                 ):
        super().__init__(env)
        env = SharedObsWrapper(env)
        if not isinstance(env, AECEnv):
            raise TypeError("The provided environment must be an instance of AECEnv")

        self.agent_ids = agent_ids or list(env.possible_agents)
        self.seed_val = None
        self.rng = np.random.default_rng(self.seed_val)
        self.comms_matrix = ActiveCommunication(self.agent_ids)
        self.failure_models = failure_models or []
        for model in self.failure_models:
            model.rng = self.rng

        if failure_models is None:
            raise ValueError("No failure model(s) provided.")
        elif isinstance(failure_models, list):
            self.failure_models = failure_models
        else:
            self.failure_models = [failure_models]

    def _update_communication_state(self):
        """
        Update the communication matrix by applying all failure models.

        This internal method applies each failure model in sequence to the communication matrix,
        determining which agents can communicate with each other for the current step.
        """
        self.comms_matrix.reset()
        for model in self.failure_models:
            model.update_connectivity(self.comms_matrix)

    def filter_action(self, agent: str, action: Any) -> Any:
        """
        Filter an agent's action if it cannot communicate with others.

        If an agent cannot communicate with any other agent, its action is replaced
        with a no-operation action appropriate for the action space.

        :param agent: The ID of the agent whose action is being filtered.
        :param action: The original action proposed by the agent.
        :return: Either the original action if communication is possible, or a no-op action if not.
        """

        can_send = any(
            self.comms_matrix.can_communicate(agent, receiver)
            for receiver in self.agent_ids
            if receiver != agent
        )
        if not can_send:
            action_space = self.env.action_space(agent)
            if hasattr(action_space, "no_op_index"):
                return action_space.no_op_index
            elif isinstance(action_space, spaces.Discrete):
                return 0
            elif isinstance(action_space, spaces.MultiDiscrete):
                return np.zeros(action_space.nvec.shape, dtype=action_space.dtype)
            elif isinstance(action_space, spaces.Box):
                return np.zeros(action_space.shape, dtype=action_space.dtype)
            else:
                return 0
        return action

    def _apply_comm_mask(self, obs: dict, receiver: str) -> dict:
        """
        Masks the part of the observation dictionary that the receiver
        should not see based on the communication matrix.

        :param obs: The original observation dictionary.
        :param receiver: The ID of the agent receiving the observation.
        :return: The masked observation dictionary where information from agents that cannot
            communicate with the receiver is zeroed out.
        """
        # Early exit if agents can communicate with receiver
        if all(
            self.comms_matrix.can_communicate(sender, receiver) or sender == receiver
            for sender in self.agent_ids
        ):
            return obs

        # Otherwise, apply masking
        for sender in self.agent_ids:
            if sender != receiver and not self.comms_matrix.can_communicate(
                    sender, receiver
            ):
                if sender in obs:
                    obs[sender] = np.zeros_like(obs[sender])
        return obs

    def step(self, action: Any):
        """Take a step in the environment"""
        current_agent = self.env.agent_selection
        if self.env.terminations.get(current_agent, False) or self.env.truncations.get(current_agent, False):
            self.env.step(None)  # Ensure the environment gets None action
            return
        filtered_action = self.filter_action(current_agent, action)
        self.env.step(filtered_action)
        self._update_communication_state()

    def reset_rng(self, seed=None):
        self.seed_val = seed
        self.rng = np.random.default_rng(self.seed_val)

    def reset(self, seed=None, options=None):
        """Reset the environment and communication state."""
        self.reset_rng(seed)
        result = self.env.reset(seed=seed, options=options)
        self.comms_matrix.reset()
        self._update_communication_state()

        for model in self.failure_models:
            model.rng = self.rng
        return result

    def observe(self, agent: str):

        self._update_communication_state()
        raw_obs = self.env.observe(agent)

        if isinstance(raw_obs, dict):
            raw_obs = self._apply_comm_mask(raw_obs, agent)

        if hasattr(self, "failure_models"):
            for model in self.failure_models:
                if hasattr(model, "get_corrupted_obs"):
                    raw_obs = model.get_corrupted_obs(agent, raw_obs)

        return raw_obs

    def last(self, observe: bool = True):

        self._update_communication_state()

        agent = self.env.agent_selection
        if agent not in self.env.agents:
            return None, 0.0, True, False, {}
        obs, rew, term, trunc, info = self.env.last(observe)
        if not observe:
            return None, rew, term, trunc, info

        current_agent = self.env.agent_selection
        if isinstance(obs, dict):
            obs = self._apply_comm_mask(obs, current_agent)

        if hasattr(self, "failure_models"):
            for model in self.failure_models:
                if hasattr(model, "get_corrupted_obs"):
                    obs = model.get_corrupted_obs(agent, obs)

        return obs, rew, term, trunc, info

    def get_communication_state(self):

        return self.comms_matrix.get_state()

    def add_failure_model(self, model: CommunicationModels):
        """User can add new failure model to the environment"""
        model.rng = self.rng
        self.failure_models.append(model)
