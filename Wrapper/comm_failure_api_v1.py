import numpy as np
from pettingzoo import AECEnv
from gymnasium import spaces
import gymnasium as gym
from typing import Dict, Optional, Union, List, Tuple, Any, Callable
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
        if matrix_shape != (num_agents, num_agents):
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
        self.matrix[sender_idx][receiver_idx] = can_communicate

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
    def apply(self, obs:Any):
        """Appy noise to an Observation."""
        pass

class GaussianNoise(NoiseModel):
    def __init__(self, mean: float = 0.0, std: float = 0.1, rng=None):
        super().__init__(rng)
        self.mean = mean
        self.std = std

    def apply(self, obs:Any) -> Any:
        if isinstance(obs, np.ndarray):
            noise = self.rng.normal(self.mean, self.std, size=obs.shape)
            return obs + noise

        elif isinstance(obs, dict):
            noisy_obs = {}
            for k, v in obs.items():
                if isinstance(v, np.ndarray):
                    noise = self.rng.normal(self.mean, self.std, size=v.shape)
                    noisy_obs[k] = v + noise
                elif isinstance(v, (int, float, np.integer)):
                    noise = self.rng.normal(self.mean, self.std)
                    noisy_obs[k] = v + noise
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

    def apply(self, obs:Any) -> Any:
        if isinstance(obs, np.ndarray):
            noise = self.rng.laplace(self.loc, self.scale, size=obs.shape)
            return obs + noise

        elif isinstance(obs, dict):
            noisy_obs = {}
            for k, v in obs.items():
                if isinstance(v, np.ndarray):
                    noise = self.rng.laplace(self.loc, self.scale, size=v.shape)
                    noisy_obs[k] = v + noise
                elif isinstance(v, (int, float, np.integer)):
                    noise = self.rng.laplace(self.loc, self.scale)
                    noisy_obs[k] = v + noise
                else:
                    noisy_obs[k] = v
            return noisy_obs
        else:
            return obs

class CustomNoise(NoiseModel):
    def __init__(self, noise_fn: Callable[[Any], Any], rng=None):
        super().__init__(rng)
        self.noise_fn = noise_fn
        if not callable(noise_fn):
            raise ValueError("Noise function must be callable")

    def apply(self, obs:Any) -> Any:
        return self.noise_fn(obs)

class CommunicationFailure(BaseWrapper):
    def __init__(self,
                 env: AECEnv[AgentID, ActionType, ObsType],
                 failure_models: List[CommunicationModels] = None,
                 noise_model: NoiseModel = None,
                 agent_ids: Optional[List[str]] = None,
                 seed: Optional[int] = None):
        super().__init__(env)

        if not isinstance(env, AECEnv):
            raise TypeError("The provided environment must be an instance of AECEnv")

        self.agent_ids = agent_ids or list(env.possible_agents)
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.comms_matrix = ActiveCommunication(self.agent_ids)

        # set default models if User provided none
        self.failure_models = failure_models or  [ProbabilisticModel(self.agent_ids,
                                                                     failure_prob=0.3, seed=seed)]
        self.noise_model = noise_model or GaussianNoise(0.0, 0.1, self.rng)

        for model in self.failure_models:
            model.rng = self.rng
        self.noise_model.rng = self.rng

    def _update_communication_state(self):
        """Update the communication matrix by applying all failure models"""
        self.comms_matrix.reset()
        for model in self.failure_models:
            model.update_connectivity(self.comms_matrix)

    def modify_observation(self, agent: str, raw_obs: Any) -> Any:
        """Modify an agent's observation based on communication failures"""
        communication_failures = any(
            not self.comms_matrix.can_communicate(sender, agent)
            for sender in self.agent_ids
            if sender != agent
        )
        if communication_failures:
            return self.noise_model.apply(raw_obs)
        else:
            return raw_obs

    def filter_action(self, agent: str, action: Any) ->Any:
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
        filtered_action = self.filter_action(current_agent, action)
        self.env.step(filtered_action)
        self._update_communication_state()
        obs, rew, term, trunc, info = self.env.last()
        modified_obs = self.modify_observation(current_agent, obs)
        return modified_obs, rew, term, trunc, info

    def reset(self, seed=None, options=None):
        """Reset the environment and communication state."""
        self.env.reset(seed=seed, options=options)
        self.comms_matrix.reset()
        if seed is not  None:
            self.rng = np.random.default_rng(seed)
            for model in self.failure_models:
                model.rng = self.rng
            self.noise_model.rng = self.rng

        initial_agent = self.env.agent_selection
        initial_obs = self.env.observe(initial_agent)
        modified_initial_obs = self.modify_observation(initial_agent, initial_obs)
        infos = {}
        return modified_initial_obs, infos

    def observe(self, agent):
        raw_obs = self.env.observe(agent)
        return self.modify_observation(agent, raw_obs)

    def last(self, observe: bool = True) -> tuple[ObsType | None, float, bool, bool, dict[str, Any]]:
        obs, rew, term, trunc, info = self.env.last()
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







