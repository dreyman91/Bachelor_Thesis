from pettingzoo.utils import BaseWrapper
from pettingzoo import AECEnv
from gymnasium import spaces
import numpy as np
import gymnasium as gym
from pettingzoo.utils.env import AgentID, ObsType, ActionType
from typing import Dict, Optional, Union, List, Tuple, Any, Callable, cast
import warnings
from Failure_API.src.wrapper_api.models.active_communication import ActiveCommunication
from Failure_API.src.wrapper_api.models.communication_model import CommunicationModels



class CommunicationWrapper(BaseWrapper):
    def __init__(self,
                 env: AECEnv[AgentID, ActionType, ObsType],
                 failure_models: List[CommunicationModels] = None,
                 agent_ids: Optional[List[str]] = None
                 ):
        super().__init__(env)

        if not isinstance(env, AECEnv):
            raise TypeError("The provided environment must be an instance of AECEnv")

        self.agent_ids = agent_ids or list(env.possible_agents)
        self.seed_val = None
        self.rng = np.random.default_rng(self.seed_val)
        self.comms_matrix = ActiveCommunication(self.agent_ids)
        self.failure_models = failure_models or []
        for model in self.failure_models:
            model.rng = self.rng

    def _update_communication_state(self):
        """Update the communication matrix by applying all failure models"""
        self.comms_matrix.reset()
        for model in self.failure_models:
            model.update_connectivity(self.comms_matrix)

    def filter_action(self, agent: str, action: Any) -> Any:
        """Filter an Agent's action if it cannot communicate with others. """

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

    def _apply_comm_mask(self,  obs: dict, receiver: str)->dict:
        """
        Masks the part of the Obs dictionary that the receiver
        should not see based on the communication matrix.
        """
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
        raw_obs = self.env.observe(agent)
        if isinstance(raw_obs, dict):
            raw_obs = self._apply_comm_mask(raw_obs, agent)
        return raw_obs

    def last(self, observe: bool = True):
        obs, rew, term, trunc, info = self.env.last(observe)
        if not observe:
            return None, rew, term, trunc, info

        current_agent = self.env.agent_selection
        if isinstance(obs, dict):
            obs = self._apply_comm_mask(obs, current_agent)

        return obs, rew, term, trunc, info

    def get_communication_state(self):
        return self.comms_matrix.get_state()

    def add_failure_model(self, model: CommunicationModels):
        """User can add new failure model to the environment"""
        model.rng = self.rng
        self.failure_models.append(model)