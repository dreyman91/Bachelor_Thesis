
from pettingzoo.utils.env import AgentID, AECEnv, ActionType, ObsType
import numpy as np
import copy
from gymnasium import spaces
from typing import Optional, List, Any, Dict
from .base_wrapper import BaseWrapper
from ..communication_models.active_communication import ActiveCommunication
from ..communication_models.base_communication_model import CommunicationModels
from .sharedobs_wrapper import SharedObsWrapper
from pettingzoo.utils.env import AECEnv


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
                 agent_ids: Optional[List[str]] = None,
                 verbose: bool = False,
                 ):
        env = SharedObsWrapper(env)
        super().__init__(env)
        self.env = env

        if not isinstance(env, AECEnv):
            raise TypeError("The provided environment must be an instance of AECEnv")

        if not hasattr(env, "reset") or not callable(getattr(env, "reset")):
            raise ValueError("Provided env does not have a callable reset() method.")

        self.agent_ids = agent_ids or list(env.possible_agents)
        self.comms_matrix = ActiveCommunication(self.agent_ids)
        self.failure_models = failure_models or []
        for model in self.failure_models:
            model.rng = self.rng

        self.verbose = verbose
        self._step_counter = 0
        self._last_update_step = -1
        self._agent_index_cache = {}
        self._masked_obs_cache = {}
        self._zero_cache = {}
        if failure_models is None:
            raise ValueError("No failure model(s) provided.")
        elif isinstance(failure_models, list):
            self.failure_models = failure_models
        else:
            self.failure_models = [failure_models]



    @property
    def agents(self):
        return self.env.agents

    @property
    def agents(self):
        return list(self.env.agents)

    def _update_communication_state(self):
        """
        Update the communication matrix by applying all failure models.

        This internal method applies each failure model in sequence to the communication matrix,
        determining which agents can communicate with each other for the current step.

        The communication state is only updated once per step cycle
        """
        agent_list = list(self.env.agents)
        if self._last_update_step < self._step_counter:
            self.comms_matrix.reset()
            for model in self.failure_models:
                model.update_connectivity(self.comms_matrix)
            self._last_update_step = self._step_counter

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

    def _apply_comm_mask(self, obs: Dict[str, np.ndarray], agent: str, output_dict=None) -> Dict[str, np.ndarray]:
        """Apply communication mask to the observation dictionary for a specific agent.

        Args:
            obs: Original observation dictionary
            agent: Agent ID receiving the observation
            output_dict: Optional pre-allocated dictionary to reuse
        """
        if output_dict is None:
            output_dict = {}
        output_dict.clear()

        # Cache matrix lookup
        if not hasattr(self, '_last_matrix_step'):
            self._last_matrix_step = -1
            self._comm_mask_cache = {}

        # Only recalculate masks when matrix changes
        if self._last_matrix_step != self._last_update_step:
            self._comm_mask_cache.clear()
            self._last_matrix_step = self._last_update_step

        # Get cached mask
        if agent not in self._comm_mask_cache:
            if agent not in self._agent_index_cache:
                self._agent_index_cache[agent] = self.agent_ids.index(agent)
            agent_index = self._agent_index_cache[agent]

            # Cache the mask for this agent
            comm_matrix = self.comms_matrix.get_boolean_matrix()
            self._comm_mask_cache[agent] = comm_matrix[:, agent_index]

        comm_mask = self._comm_mask_cache[agent]

        # Use precomputed indices for faster iteration
        if not hasattr(self, '_sender_indices'):
            self._sender_indices = {sender: i for i, sender in enumerate(self.agent_ids)}

        # Direct mask application without extra allocations
        # MISSING LOOP: Need to iterate through all potential senders
        for sender, sender_obs in obs.items():
            if sender in self._sender_indices:
                sender_index = self._sender_indices[sender]
                if sender == agent or comm_mask[sender_index]:
                    output_dict[sender] = sender_obs

        return output_dict

    def step(self, action: Any):
        """Step function compatible with aec_to_parallel and PettingZoo cycle rules."""
        current_agent = self.env.agent_selection

        self._update_communication_state()

        # Check if this agent is terminated, but still required to complete the cycle
        is_dead = self.env.terminations.get(current_agent, False) or self.env.truncations.get(current_agent, False)

        if is_dead:
            # Step with None to preserve agent cycle — do NOT skip
            if self.verbose:
                print(f"⚠️ Agent {current_agent} is already done. Stepping with None.")
            self.env.step(None)
        else:
            # Apply communication logic as usual
            filtered_action = self.filter_action(current_agent, action)
            self.env.step(filtered_action)

        self._step_counter += 1

    def reset_rng(self, seed=None):
        self.seed_val = seed
        self.rng = np.random.default_rng(self.seed_val)

    def reset(self, seed=None, options=None):
        """Reset the environment and communication state."""

        self._step_counter = 0
        self._last_update_step = -1
        self.reset_rng(seed)
        result = self.env.reset(seed=seed, options=options)
        self.comms_matrix.reset()

        for model in self.failure_models:
            model.rng = self.rng

        self._update_communication_state()

        # If a result is None, create proper return values
        if result is None:
            observations = {agent: None for agent in self.env.agents}
            infos = {agent: {} for agent in self.env.agents}
            return observations, infos
        return result

    def observe(self, agent: str) -> Dict[str, np.ndarray]:
        """Override to apply communication state masking with batch optimization."""
        # Reuse the observation dictionary
        shared_obs = self.env.observe(agent)

        # If not a dictionary, something went wrong - can't perform masking
        if not isinstance(shared_obs, dict):
            raise TypeError(f"Expected dictionary from SharedObsWrapper, got {type(shared_obs)}")

        # Filter to valid agents
        raw_obs = {
            sender: vec
            for sender, vec in shared_obs.items()
            if sender in self.agent_ids and sender in self.env.agents
        }

        # Apply communication mask
        if agent not in self._masked_obs_cache:
            self._masked_obs_cache[agent] = {}

        return self._apply_comm_mask(raw_obs, agent, self._masked_obs_cache[agent])

    def last(self, observe: bool = True):

        self._update_communication_state()

        agent = self.env.agent_selection
        if agent not in self.env.agents:
            return None, 0.0, True, False, {}
        _, rew, term, trunc, info = self.env.last(observe)
        if not observe:
            return None, rew, term, trunc, info

        obs = self.observe(agent)
        return obs, rew, term, trunc, info

    def get_communication_state(self):
        self._update_communication_state()
        return self.comms_matrix.get_state()

    def add_failure_model(self, model: CommunicationModels):
        """User can add new failure model to the environment"""
        model.rng = self.rng
        self.failure_models.append(model)
