

"""
This CommunicationFailure API functions as a wrapper for simulating communication failures
in  multi-agent reinforcement learning (MARL) environments using the PettingZoo Framework. It allows
the User to define various communication_models for how communication between agents can be disrupted(e.g distanced-based,
probabilistic) and how observations can be corrupted by noise(Gaussian, Laplacian). This is crucial for training
training robust agents that can perform well under imperfect communication conditions.

- class ActiveCommunication:- Manages the current state of communication links or maintains the communication 
matrix that defines which agents can talk together. Allows updates when failures occur, blocks communication 
between some agents, and can reset to a fully connected state.

    - _validate_matrix: a static method that checks if the provided matrix is valid.
    
    - can_communicate: checks if sender can communicate with receiver.
    - update: modifies the communication state between agents.
    - blocked_agents: gets a list of agents that cannot communicate with the given agent.
    - reset: resets the communication matrix to full connectivity.
    - get_state: returns full state of the communication matrix.
    

- class CommunicationModels(Abstract): A base class that defines how failures can happen. Subclasses 
extend this to implement specific failure logic.
    - DistanceModel:  Communication is defined by a distance threshold.
    - ProbabilisticModel:  Communication is randomly disrupted based on a fixed failure probability.
    - MatrixModel: Predefined communication matrix is used to define which agents can talk together,
                   allowing custom control over failures.
                   
    - update_connectivity: modifies the communication matrix to reflect the communication failure
                           defined by that model.


- class NoiseModel(Abstract): Base class to for defining how noise is used in adding uncertainty to 
agent observations.
    - Gaussian Noise: Adds normally distributed noise to observations
    - Laplacian Noise: Adds Laplacian (spike-like) noise to observations      
    - CustomNoise: Allows User to define their own noise function.
    
- class CommunicationFailure: The Primary or main class that applies the communication failure and noise
communication_models to the environment.

:methods
        - _update_communication: resets the communication matrix to fully connected, iterates over all failure 
        communication_models and applies them, then updates the matrix to reflect new communication failures.
        
        - modify_observation: It checks if an agent is affected by communication failures. if communication is 
        blocked or affected, it applies noise distortion to the agent's observation and, returns a noisy or unmodified
        observation.
        
        - filter_action: Checks if an agent can communicate with others through the can_communicate method,
        if the agent can not send, it returns the NO-OP action. if agent can, it returns the original action.
        
        - step: The API intercepts the step method of the wrapped environment.
                - determines the current agent (from agent_selection)
                - check if agent is done (terminated or truncated)
                - filters the provided action based on communication state
                - calls step on the wrapped environment
                - updates the communication state for the next step
                
        - reset: The API overrides the reset method of the wrapped environment.
                - resets the underlying environment.
                - resets the the matrix to full connectivity
                - updates the communication state (applies failure communication_models)
                - handles seeding for reproducibility.
                - gets the initial observation for first agent, applies noise, and modifies the observation
                  based on initial communication failures.
                - returns the noise and failure modified observation.
                
        - observe: The API overrides the observe method of the wrapped environment.
                - retrieves the raw observation of the current agant from the underlying environment
                - applies the noise model and, modifies the noisy observation based on communication failures.
                - returns the modified observation.
        
        - last: The API overrides the last method of the wrapped environment.
                - get the observation, reward, termination, truncation and info for the current agent.
                - if observe is true , returns the noise and modified observation.
                
        - observation_space: Returns the current agent's observation space from the underlying environment.
        
        - action_space: Returns the current agent's action space from the underlying environment.
        - get_communication_state: Returns a copy of the communication matrix state.
        - add_failure_model: allows the addition of communication failure communication_models after initialization.
        - set_noise_model: allows changing the noise model after initialization.            

"""

import numpy as np
from pettingzoo.utils import BaseWrapper
from typing import Optional

class BaseWrapper(BaseWrapper):
    """Shared functionality (e.g. seeding)."""
    def __init__(self, env, seed: Optional[int] = None):
        super().__init__(env)
        self.seed_val = seed
        self.rng = np.random.default_rng(seed)

    def reset_rng(self, seed: Optional[int] = None):
        """Update RNG and propagate to internal communication_models."""
        self.seed_val = seed
        self.rng = np.random.default_rng(seed)