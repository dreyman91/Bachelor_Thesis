import numpy as np
from pettingzoo import AECEnv
from gymnasium import spaces
import gymnasium as gym
from typing import Dict, Optional, Union, List, Tuple, Any, Callable, cast
import warnings
from pettingzoo.utils import BaseWrapper
from pettingzoo.utils.env import ActionType, ObsType, AgentID
from abc import ABC, abstractmethod


class ActiveCommunication:
    """Represents the actual communication state between agents."""

    def __init__(self,
                 agent_ids: List[str],
                 initial_matrix: Optional[np.ndarray] = None):
        self.agent_ids = agent_ids
        self.agent_id_to_index = {agent: i for i, agent in enumerate(self.agent_ids)}
        num_agents = len(agent_ids)

        if initial_matrix is not None:
            self._validate_matrix(initial_matrix, num_agents)
            self.matrix = initial_matrix.astype(bool)
        else:
            self.matrix = np.ones((num_agents, num_agents))

    @staticmethod
    def _validate_matrix(matrix, num_agents):
        if matrix.shape != (num_agents, num_agents):
            raise ValueError(f"Matrix must be {num_agents} x {num_agents}")
        if not np.all(np.diag(matrix) == 1):
            raise ValueError(f"Self Communication must always be enabled")
        if not np.all((matrix >= 0) & (matrix <= 1)):
            raise ValueError(f"Matrix must be either 0 or 1")

    def can_communicate(self, sender: str, receiver: str) -> bool:
        """Returns True if the agent can communicate with other agents."""
        sender_idx = self.agent_id_to_index[sender]
        receiver_idx = self.agent_id_to_index[receiver]
        return bool(self.matrix[sender_idx, receiver_idx])

    def update(self, sender: str, receiver: str, can_communicate: bool):
        """Updates communication status"""
        sender_idx = self.agent_id_to_index[sender]
        receiver_idx = self.agent_id_to_index[receiver]
        self.matrix[receiver_idx, sender_idx] = can_communicate

    def get_blocked_agents(self, agent: str) -> List[str]:
        """Returns a list of agents that cannot communicate with the given agent."""
        agent_idx = self.agent_id_to_index[agent]
        blocked_indices = np.where(self.matrix[agent_idx] == False)[0]
        blocked_agents = [self.agent_ids[i] for i in blocked_indices if self.agent_ids[i] != agent]
        return blocked_agents

    def reset(self):
        """Resets the communication matrix to fully connected"""
        self.matrix = np.ones((len(self.agent_ids), len(self.agent_ids)), dtype=bool)

    def get_state(self) -> np.ndarray:
        """Gets a copy of the current state"""
        return self.matrix.copy()


class CommunicationModels(ABC):
    """Abstract base class for communication models."""

    @abstractmethod
    def update_connectivity(self, comms_matrix: ActiveCommunication):
        """Updates the ActiveCommunication and inject failures"""
        pass

    @staticmethod
    @abstractmethod
    def create_initial_matrix(agent_ids: List[str]) -> np.ndarray:
        """Creates the initial ActiveCommunication matrix before failures"""
        pass


class DistanceModel(CommunicationModels):
    def __init__(self, agent_ids: List[str], distance_threshold: float, pos_fn: Callable[[], Dict[str, np.ndarray]],
                 failure_prob: float = 0.0, seed: Optional[int] = None):
        self.agent_ids = agent_ids
        self.distance_threshold = distance_threshold
        self.pos_fn = pos_fn
        self.failure_prob = failure_prob
        self.rng = np.random.default_rng(seed)

    def update_connectivity(self, comms_matrix: ActiveCommunication):
        positions = self.pos_fn()
        for sender in comms_matrix.agent_ids:
            for receiver in comms_matrix.agent_ids:
                if sender != receiver:
                    dist = np.linalg.norm(positions[sender] - positions[receiver])
                    can_communicate = dist <= self.distance_threshold

                    if can_communicate and self.rng.random() < self.failure_prob:
                        can_communicate = False
                    comms_matrix.update(sender, receiver, can_communicate)

    @staticmethod
    def create_initial_matrix(agent_ids: List[str]) -> np.ndarray:
        return np.ones((len(agent_ids), len(agent_ids)), dtype=bool)


class ProbabilisticModel(CommunicationModels):
    def __init__(self, agent_ids: List[str], failure_prob: float, seed: Optional[int] = None):
        self.agent_ids = agent_ids
        self.failure_prob = failure_prob
        self.rng = np.random.default_rng(seed)

    def update_connectivity(self, comms_matrix: ActiveCommunication):
        for sender in comms_matrix.agent_ids:
            for receiver in comms_matrix.agent_ids:
                if sender != receiver:
                    if self.rng.random() < self.failure_prob:
                        comms_matrix.update(sender, receiver, False)
    @staticmethod
    def create_initial_matrix(agent_ids: List[str]) -> np.ndarray:
        return np.ones((len(agent_ids), len(agent_ids)), dtype=bool)


class MatrixModel(CommunicationModels):
    def __init__(self, agent_ids: List[str], comms_matrix_values: np.ndarray, failure_prob: float = 0.0,
                 seed: Optional[int] = None):
        self.agent_ids = agent_ids
        self.comms_matrix_values = comms_matrix_values
        self.failure_prob = failure_prob
        self.rng = np.random.default_rng(seed)

    def update_connectivity(self, comms_matrix: ActiveCommunication):
        for sender in comms_matrix.agent_ids:
            for receiver in comms_matrix.agent_ids:
                if sender != receiver:
                    sender_idx = comms_matrix.agent_id_to_index[sender]
                    receiver_idx = comms_matrix.agent_id_to_index[receiver]
                    initial_connectivity = self.comms_matrix_values[sender_idx, receiver_idx] == 1

                    if initial_connectivity and self.rng.random() < self.failure_prob:
                        comms_matrix.update(sender, receiver, False)

    @staticmethod
    def create_initial_matrix(agent_ids: List[str]) -> np.ndarray:
        return np.array([], dtype=bool)


class NoiseModel(ABC):
    """Base class for noise models."""

    def __init__(self, rng=None):
        self.rng = rng or np.random.default_rng()

    def set_rng(self, rng):
        """Set random number generator."""
        self.rng = rng

    @abstractmethod
    def apply(self, obs: Any):
        """Appy noise to an Observation."""
        pass


class GaussianNoise(NoiseModel):
    def __init__(self, mean: float = 0.0, std: float = 0.0, rng=None):
        super().__init__(rng)
        self.mean = mean
        self.std = std

    def apply(self, obs: Any, observation_space: Optional[spaces.Space] = None) -> Any:
        if isinstance(obs, np.ndarray):
            if observation_space is not None and isinstance(observation_space, spaces.Box):
                noise = self.rng.normal(loc=self.mean, scale=self.std, size=obs.shape)
                noisy_obs = obs + noise
                noisy_obs = np.clip(noisy_obs, observation_space.low, observation_space.high)
                return noisy_obs

            else: # if not known Box space, just add noise
                return obs + self.rng.normal(loc=self.mean, scale=self.std, size=obs.shape)

        elif isinstance(obs, dict):
            noisy_obs = {}
            for k, v in obs.items():
                if isinstance(v, np.ndarray):
                    if observation_space and k in observation_space and isinstance(observation_space[k], spaces.Box):
                        noise = self.rng.normal(loc=self.mean, scale=self.std, size=v.shape)
                        noise_arr = v + noise
                        noise_arr = np.clip(noise_arr, observation_space[k].low, observation_space[k].high)
                        noisy_obs[k] = noise_arr
                    else:  # No Observation space, or not a box
                        noisy_obs[k] = v + self.rng.normal(self.mean, self.std, size=v.shape)

                elif isinstance(v, (int, float, np.integer)):  # Handle Discrete and MultiDiscrete
                    if observation_space and k in observation_space:
                        if isinstance(observation_space[k], spaces.Discrete):
                            noise_val = v + self.rng.normal(self.mean, self.std)
                            clipped_val = np.clip(noise_val, 0, observation_space[k].n - 1)
                            noisy_obs[k] = int(round(clipped_val))
                        elif isinstance(observation_space[k], spaces.MultiDiscrete):
                            noise_val = v + self.rng.normal(self.mean, self.std)
                            clipped_val = np.clip(noise_val, 0, observation_space[k].nvec - 1)
                            noisy_obs[k] = np.round(clipped_val).astype(int)
                        elif isinstance(observation_space[k], spaces.Box):
                            noise = self.rng.normal(self.mean, self.std)
                            noise_arr = float(v) + noise
                            noise_arr = np.clip(noise_arr, observation_space[k].low, observation_space[k].high)
                            noisy_obs[k] = noise_arr
                        else:
                            noisy_obs[k] = v + self.rng.normal(self.mean, self.std)
                else:
                    noisy_obs[k] = v
            return noisy_obs
        else:
            return obs


class LaplacianNoise(NoiseModel):
    """Applies Laplacian Noise to the Observation"""
    def __init__(self, loc: float = 0.0, scale: float = 0.1, rng=None):
        super().__init__(rng)
        self.loc = loc
        self.scale = scale

    def apply(self, obs: Any, observation_space: Optional[spaces.Space] = None) -> Any:
        if isinstance(obs, np.ndarray):
            if observation_space is not None and isinstance(observation_space, spaces.Box):
                noise = self.rng.laplace(loc=self.loc, scale=self.scale, size=obs.shape)
                noisy_obs = obs + noise
                noisy_obs = np.clip(noisy_obs, observation_space.low, observation_space.high)
                return noisy_obs
            else:  # if not known Box space, just add noise
                return obs + self.rng.laplace(loc=self.loc, scale=self.scale, size=obs.shape)

        elif isinstance(obs, dict):
            noisy_obs = {}
            for k, v in obs.items():
                if isinstance(v, np.ndarray):
                    if observation_space and k in observation_space and isinstance(observation_space[k], spaces.Box):
                        noise = self.rng.laplace(loc=self.loc, scale=self.scale, size=v.shape)
                        noise_arr = v + noise
                        noise_arr = np.clip(noise_arr, observation_space[k].low, observation_space[k].high)
                        noisy_obs[k] = noise_arr
                    else:  # No Observation space, or not a box
                        noisy_obs[k] = v + self.rng.laplace(self.loc, self.scale, size=v.shape)

                elif isinstance(v, (int, float, np.integer)):  # Handle Discrete and MultiDiscrete
                    if observation_space and k in observation_space:
                        if isinstance(observation_space[k], spaces.Discrete):
                            noise_val = v + self.rng.laplace(self.loc, self.scale)
                            clipped_val = np.clip(noise_val, 0, observation_space[k].n - 1)
                            noisy_obs[k] = int(round(clipped_val))
                        elif isinstance(observation_space[k], spaces.MultiDiscrete):
                            noise_val = v + self.rng.laplace(self.loc, self.scale)
                            clipped_val = np.clip(noise_val, 0, observation_space[k].nvec - 1)
                            noisy_obs[k] = np.round(clipped_val).astype(int)
                        elif isinstance(observation_space[k], spaces.Box):
                            noise = self.rng.laplace(self.loc, self.scale)
                            noise_arr = float(v) + noise
                            noise_arr = np.clip(noise_arr, observation_space[k].low, observation_space[k].high)
                            noisy_obs[k] = noise_arr
                        else:
                            noisy_obs[k] = v + self.rng.laplace(self.loc, self.scale)
                else:
                    noisy_obs[k] = v
            return noisy_obs
        else:
            return obs


class CustomNoise(NoiseModel):
    def __init__(self, noise_fn: Callable[[Any, Optional[spaces.Space]], Any], rng=None):
        super().__init__(rng)
        self.noise_fn = noise_fn
        if not callable(noise_fn):
            raise ValueError("Noise function must be callable")

    def apply(self, obs: Any, observation_space: Optional[spaces.Space] = None) -> Any:
        noisy_obs = self.noise_fn(obs, observation_space)
        return noisy_obs


class CommunicationFailure(BaseWrapper):
    def __init__(self,
                 env: AECEnv[AgentID, ActionType, ObsType],
                 failure_models: List[CommunicationModels] = None,
                 noise_model: NoiseModel = None,
                 agent_ids: Optional[List[str]] = None
                 ):
        super().__init__(env)

        if not isinstance(env, AECEnv):
            raise TypeError("The provided environment must be an instance of AECEnv")

        self.agent_ids = agent_ids or list(env.possible_agents)
        self.seed_val = None
        self.rng = np.random.default_rng(self.seed_val)
        self.comms_matrix = ActiveCommunication(self.agent_ids)

        # set default models if User provided none
        self.failure_models = failure_models or [ProbabilisticModel(self.agent_ids,
                                                                    failure_prob=0.3, seed=self.seed_val)]
        self.noise_model = noise_model or GaussianNoise(0.0, 0.1, self.rng)

        for model in self.failure_models:
            model.rng = self.rng
        self.noise_model.rng = self.rng

    def _update_communication_state(self):
        """Update the communication matrix by applying all failure models"""
        self.comms_matrix.reset()
        for model in self.failure_models:
            model.update_connectivity(self.comms_matrix)

    def modify_observation(self, agent: str, noisy_obs: Any) -> Any:
        """Modify an agent's observation based on communication failures"""
        communication_failures = any(
            not self.comms_matrix.can_communicate(sender, agent)
            for sender in self.agent_ids
            if sender != agent
        )
        if communication_failures:
            return noisy_obs
        else:
            return noisy_obs

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

    def step(self, action: Any):
        """Take a step in the environment"""
        current_agent = self.env.agent_selection
        if self.env.terminations.get(current_agent, False) or self.env.truncations.get(current_agent, False):
            self.env.step(None)  # Ensure the environment gets None action
            return
        filtered_action = self.filter_action(current_agent, action)
        self.env.step(filtered_action)
        self._update_communication_state()

    def reset(self, seed=None, options=None):
        """Reset the environment and communication state."""
        self.env.reset(seed=seed, options=options)
        self.comms_matrix.reset()
        self._update_communication_state()
        if seed is not None:
            self.seed_val = seed
            self.rng = np.random.default_rng(self.seed_val)
            for model in self.failure_models:
                model.rng = self.rng
            self.noise_model.rng = self.rng

        initial_agent = self.env.agent_selection
        initial_obs = self.env.observe(initial_agent)
        modified_initial_obs = self.modify_observation(initial_agent, initial_obs)
        infos = {}
        return modified_initial_obs, infos

    def seed(self, seed=None):
        self.seed_val = seed
        self.reset(seed=seed)

    def observe(self, agent):
        raw_obs = self.env.observe(agent)
        observation_space = self.env.observation_space(agent)
        noisy_obs = self.noise_model.apply(raw_obs, observation_space)
        return self.modify_observation(agent, noisy_obs)

    def last(self, observe: bool = True) -> tuple[ObsType | None, float, bool, bool, dict[str, Any]]:
        obs, rew, term, trunc, info = self.env.last()
        if not observe:
            return None, rew, term, trunc, info
        modified_obs = self.modify_observation(self.env.agent_selection, obs)
        return modified_obs, rew, term, trunc, info

    def observation_space(self, agent):
        return self.env.observation_space(agent)

    def action_space(self, agent):
        return self.env.action_space(agent)

    def get_communication_state(self):
        return self.comms_matrix.get_state()

    def add_failure_model(self, model: CommunicationModels):
        """User can add new failure model to the environment"""
        model.rng = self.rng
        self.failure_models.append(model)

    def set_noise_model(self, model: NoiseModel):
        """Set new Noise model for the environment."""
        model.rng = self.rng
        self.noise_model = model
