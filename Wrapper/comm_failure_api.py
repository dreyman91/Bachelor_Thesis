import numpy as np
from pettingzoo import AECEnv, ParallelEnv
from gymnasium import spaces
import gymnasium as gym
from typing import Dict, Optional, Union, List, Tuple, Any, Callable
import warnings

from torch.distributed.collective_utils import all_gather_object_enforce_type



'''
This CommunicationFailure functions as a wrapper around a standard PettingZoo environment.  It dynamically modifies the 
observations of agents, to simulate different kinds of communication failures between agents. This wrapper is built on 
these concepts.

1. Communication Models
-----------------------
The API has different types of communication failure models, such as the:
- Probabilistic Model: It simulates random packet loss. Each message transmission has a user-defined probability of 
failure.
- Matrix: A user supplied defined matrix defines the probability of successful communication between every pair of agents.
Distance-Based: Communication success is determined by the spatial proximity of agents.

2. Active Communication State
----------------------------
To track the functional communication links, the API maintains an internal attribute "active_comms". It is a dictionary
dictionaries, structured as "active_comms[sender][receiver]". A value of 'True' if the sender can communicate with the 
receiver at the current timestep, "False" indicates a communication failure.

3. Observation Modification
---------------------------
The API modifies the agent observations to reflect the impact of communication failures. When an agent is unable to 
receive information from other agents due to communication failures, stochastic noise are added to the observation. This
could be a Gaussian or Laplacian with their parameters, and they  are configurable.

4. Action Filtering
------------------
Whenever an agent's outgoing communication is corrupted, it loses the ability to influence other agents. The API
modifies the "intended action" of the affected agent by replacing it with a default chosen based on the agent's space 
or a "No-Op" action. 

####----The Code----#####

class CommunicationFailure :- This is the primary class of the API

 a. __init__(...) : This constructor initializes the primary class object, setting up the internal structures

b. Parameters: 
        - env:  The base environment from PettingZoo that will be wrapped  and could either be ParallelEnv or AECEnv.
        - comms_model: The communication model.
        - comms_matrix: A Numpy array of shapes (i, j), which represents the probability of successful communication 
        between agents.
        - failure_prob: represents the probability of message failure and, only used when model 
        is set to "probabilistic"
        - agent_id: list of strings representing the IDs of agents that should be affected by communication 
        failure.
        - distance_threshold: maximum distance agents can communicate with each other. Used whom model is distance 
        based.
        - pos_fn: provides the positions of agents.
        - seed: an integer used to seed to generate random  numbers, to ensure reproducibility of communication 
        failures.
        - noise_mean: mean noise distribution for observation modification.
        - noise_std: standard deviation noise distribution for observation modification.
        - noise_distribution: a string that specifies the type of noise distribution to apply..
        - custom_noise_fn: user-defined noise addition.
        
c. _Validate_comms_matrix(matrix): A private method that performs input validation check on the provided matrix.
        - checks the shape (matrix must be square)
        - checks self communication (diagonal elements must be 1)
        - checks probability range (elements must be between 0 and 1)
        
d. _init_communication_state(): A private method that initializes the active_comms attribute, which stores the 
current state of communication links between agents.

e. _distance_based_update(): A private method used when comms_model is set to "distance".
        - It obtains agent positions by calling the user provided pos_fn to obtain a dictionary mapping agents IDs
        to their positions.
        - It calculates pairwise distances between agents.
        - It updates the active_comms for each pair of agents after determining their distance threshold.
        
f. _update_communication_state(): A private method that reflects the current state of communication links 
        between agents.
        
g. modify_observation(: modifies the observation of a given agent.
        - It checks if any agent is unable to communicate with any other agent by examining active_comms dictionary.
        -  It either adds the default noise_distribution or calls the custom_noise_fn when altering observations.
        
h. filter_action(): This method modifies the action of the given agent, based on the communication links with 
        other agents.
        - It checks if an agent has communication failure and then alters the action based off the state.
        
i. step(): the interface for performing the simulation.

j. reset(): resets the environment snd the communication dailure models to their initial state.

k. observation_space(): returns observation space for a given agent

l. action_space(): returns action space for a given agent

m. __getattr__(): this method enables "pass_through" access to attributes and methods of the wrapped environment.
        
        

'''
class CommunicationFailure:
    def __init__(self,
                 env: Union[AECEnv, ParallelEnv],
                 comms_model: str = "probabilistic",
                 comms_matrix: Optional[np.ndarray] = None,
                 failure_prob: float = 0.3,
                 agent_id: Optional[List[str]] = None,
                 distance_threshold: float = 1.0,
                 pos_fn: Optional[Callable[[], Dict[str, np.ndarray]]] = None,
                 seed: Optional[int] = None,
                 noise_mean: float = 0.0,
                 noise_std: float = 0.1,
                 noise_distribution: str = "gaussian",
                 custom_noise_fn: Optional[Callable[[Any], Any]] = None):

        self.env = env
        self.comms_model = comms_model
        self.comms_matrix = comms_matrix
        self.failure_prob = failure_prob
        self.agent_id = agent_id or env.possible_agents
        self.distance_threshold = distance_threshold
        self.pos_fn = pos_fn
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.noise_mean = noise_mean
        self.noise_std = noise_std
        self.noise_distribution = noise_distribution
        self.custom_noise_fn = custom_noise_fn


        if comms_matrix is not None:
            self._validate_comms_matrix(comms_matrix)
        self.comms_matrix = comms_matrix

        self.active_comms = self._init_communication_state()
        self.is_parallel = isinstance(env, ParallelEnv)

        if comms_model == "distance" and pos_fn is None:
            raise ValueError("Distance model requires position function")

    def _validate_comms_matrix(self, matrix):
        n_agents = len(self.agent_id)
        if matrix.shape != (n_agents, n_agents):
            raise ValueError(f"Matrix must be {n_agents} x {n_agents}")
        if not np.all(np.diag(matrix) == 1):
            raise ValueError("Self-communication must always be enabled")
        if not np.all((matrix >= 0) & (matrix <= 1)):
            raise ValueError("All matrix value should be between 0 and 1")


    def _init_communication_state(self) -> Dict[str, Dict[str, bool]]:

        return{
            sender: {
                receiver: True for receiver in self.agent_id
            }
            for sender in self.agent_id
        }

    def _distance_based_update(self):
        positions = self.pos_fn()
        for agent in self.agent_id:
            if agent not in positions:
                raise ValueError(f"Position function did not return position for agent {agent}")

        for sender in self.agent_id:
            for receiver in self.agent_id:
                if sender != receiver:
                    dist = np.linalg.norm(positions[sender] - positions[receiver])
                    self.active_comms[sender][receiver] = dist <= self.distance_threshold

    def _update_communication_state(self):

        agent_indices = {a: i for i, a in enumerate(self.agent_id)}

        if self.comms_model == "matrix":
            for sender in self.agent_id:
                for receiver in self.agent_id:
                    if sender != receiver:
                        comm_prob = float(self.comms_matrix[
                            agent_indices[sender],
                            agent_indices[receiver]
                        ])
                        self.active_comms[sender][receiver] = self.rng.random() < comm_prob

        elif self.comms_model == "probabilistic":
            for sender in self.agent_id:
                for receiver in self.agent_id:
                    if sender != receiver:
                        success = self.rng.random() > self.failure_prob
                        self.active_comms[sender][receiver] = \
                            success

        elif self.comms_model == "distance":
            positions = self.pos_fn()
            for sender in self.agent_id:
                for receiver in self.agent_id:
                    if sender == receiver:
                        continue
                    dist = np.linalg.norm(
                        positions[sender] - positions[receiver]
                    )
                    self.active_comms[sender][receiver] = \
                        dist <= self.distance_threshold
        else:
            raise ValueError("Unknown Communication model: " + self.comms_model)

    def modify_observation(self, agent: str, raw_obs: Any) -> Any:
        if any(
            not self.active_comms[sender][agent]
            for sender in self.agent_id
            if sender != agent
        ):
            if self.custom_noise_fn:
                return self.custom_noise_fn(raw_obs)
            else:
                if isinstance(raw_obs, np.ndarray):
                    if self.noise_distribution == "gaussian":
                        noise = self.rng.normal(self.noise_mean, self.noise_std)
                    elif self.noise_distribution == "laplacian":
                        noise = self.rng.laplace(self.noise_mean, self.noise_std)
                    else:
                        noise = self.rng.normal(-self.noise_mean, self.noise_std)
                    return raw_obs + noise

                elif isinstance(raw_obs, dict):
                    noisy_obs = {}
                    for k, v in raw_obs.items():
                        if isinstance(v, np.ndarray):
                            if self.noise_distribution == "gaussian":
                                noise = self.rng.normal(self.noise_mean, self.noise_std)
                            elif self.noise_distribution == "laplacian":
                                noise = self.rng.laplace(self.noise_mean, self.noise_std,)
                            else:
                                noise = self.rng.normal(-self.noise_mean, self.noise_std)
                            noisy_obs[k] = v + noise

                        elif isinstance(v, (int, float, np.integer)):
                            if self.noise_distribution == "gaussian":
                                noisy_obs[k]  = v + self.rng.normal(self.noise_mean, self.noise_std, size=raw_obs.shape)
                            elif self.noise_distribution == "laplacian":
                                noisy_obs[k]  = v + self.rng.laplace(self.noise_mean, self.noise_std, size=raw_obs.shape)
                            else:
                                noisy_obs[k]  = v + self.rng.normal(-self.noise_mean, self.noise_std, size=raw_obs.shape)
                            noisy_obs[k] = v
                        else:
                            noisy_obs[k] = v
                    return noisy_obs
                else:
                    return raw_obs
        else:
            return raw_obs

    def filter_action(self, agent:str, action: Any) -> Any:

        can_send = any(
            self.active_comms[agent][receiver]
            for receiver in self.agent_id
            if receiver != agent
        )

        if not can_send:
            action_space = self.env.action_space(agent)
            if hasattr(action_space, "no_op_index"):
                return action_space.no_op_index
            elif isinstance(action_space, spaces.Discrete):
                return 0
            elif isinstance(action_space, spaces.MultiDiscrete):
                return 0
            elif isinstance(action_space, spaces.Box):
                return np.zeros(action_space.shape, dtype=action_space.dtype)
            else:
                return 0
        return action

    def step(self, actions: Dict[str, np.ndarray]):

        self._update_communication_state()

        # Filter based on communication state
        filtered_actions = {
            agent: self.filter_action(agent, action)
            for agent, action in actions.items()
        }

        ## PARALLEL vs AEC ENVIRONMENT HANDLING
        if self.is_parallel:
            observations, rewards, terminations, truncations, infos = self.env.step(filtered_actions)

        elif hasattr(self.env, "agent_iter"):
            observations = {}
            rewards = {}
            terminations = {}
            truncations = {}
            infos = {}
            for agent in self.env.agent_iter():
                obs, rew, term, trunc, info = self.env.last()

                # Store agent before stepping
                observations[agent] = obs
                rewards[agent] = rew
                terminations[agent] = term
                truncations[agent] = trunc
                infos[agent] = info

                if agent in filtered_actions and not (terminations[agent] or truncations[agent]):
                    self.env.step(filtered_actions[agent])
                else:
                    self.env.step(None)

                next_obs, _, _, _, _ = self.env.last()
                observations[agent] = next_obs

        modified_observations = {
            agent: self.modify_observation(agent, obs)
            for agent, obs in observations.items()
        }

        return modified_observations, rewards, terminations, truncations, infos

    def reset(self, seed=None, options=None):
        if self.is_parallel:
            observations, infos = self.env.reset(seed=seed, options=options)
        else:
            self.env.reset(seed=seed, options=options)
            observations = {agent: self.env.observe(agent) for agent in self.env.agents}
            infos = {}

        self.active_comms = self._init_communication_state()

        return {
            agent: self.modify_observation(agent, obs)
            for agent, obs in observations.items()
        }, infos

    def observation_space(self, agent):
        return self.env.observation_space(agent)

    def action_space(self, agent):
        return self.env.action_space(agent)

    def __getattr__(self, name):
        return getattr(self.env, name)