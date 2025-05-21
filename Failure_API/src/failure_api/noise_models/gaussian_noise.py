import numpy as np
from typing import Optional, Any, Callable, Dict, Union, Tuple
from gymnasium import spaces
from .base_noise_model import NoiseModel


class GaussianNoiseModel(NoiseModel):
    """
    A model that adds Gaussian (normally distributed) noise to observations.

    This model implements the addition of noise sampled from a normal distribution:

    X_noisy = X + N(μ, σ²)

    Where:
    - X is the original observation
    - μ is the mean of the noise distribution (typically 0)
    - σ² is the variance of the noise distribution
    - N(μ, σ²) represents a random sample from a normal distribution

    The model handles different observation types (arrays, dictionaries, scalars)
    and respects the bounds of gymnasium observation spaces when provided.
    """

    def __init__(self, mean: float = 0.0,
                 std: float = 0.0,
                 ):
        """
        Initialize the Gaussian noise model.

        Parameters:
        mean : Mean of the Gaussian noise distribution
        std : Standard deviation of the Gaussian noise distribution
        """
        super().__init__()
        self.mean = mean
        self.std = std

    def apply(self, obs: Any, observation_space: Optional[spaces.Space] = None) -> Any:
        """
        Apply Gaussian noise to an observation.

        This method handles different observation types and respects
        the bounds of observation spaces when provided.

        Parameters:
        obs : Observation to which noise will be added
        observation_space : Space defining the valid range for the observation

        Returns: Observation with added Gaussian noise
        """
        if self.rng is None:
            raise ValueError("Random number generator not initialized.")
        if isinstance(obs, np.ndarray):
            return self._apply_to_array(obs, observation_space)
        elif isinstance(obs, dict):
            return self._apply_to_dict(obs, observation_space)
        elif isinstance(obs, (int, float, np.integer, np.floating)):
            return self._apply_to_scalar(obs, observation_space)
        else:
            # for unsupported types
            return obs

    def _apply_to_array(self,
                        array: np.ndarray,
                        space: Optional[spaces.Space] = None) -> np.ndarray:
        """
        Apply Gaussian noise to a numpy array observation.

        :param array: Array observation
        :param space: Space defining valid range for the array
        :return:  with added Gaussian noise
        """
        # Generate noise of the same shape as the input array
        noise = self.rng.normal(loc=self.mean, scale=self.std, size=array.shape)

        # Add noise to Observation
        noisy_array = array + noise

        # Clip to space bounds if provided
        if space is not None and isinstance(space, spaces.Box):
            noisy_array = np.clip(noisy_array, space.low, space.high)
        return noisy_array

    def _apply_to_dict(self,
                       dict_obs: Dict,
                       dict_space: Optional[Dict[str, spaces.Space]] = None) -> Dict:
        """
        Apply Gaussian noise to a dictionary observation.
        :param dict_obs: Dictionary observation
        :param dict_space: Dictionary of spaces defining valid ranges
        :return: Dictionary with added Gaussian noise
        """

        noisy_obs = {}

        for key, value in dict_obs.items():
            # Get space key if available
            key_space = None
            if dict_space is not None and isinstance(dict_space, spaces.Dict) and key in dict_space.spaces:
                key_space = dict_space.spaces[key]
            elif dict_space is not None and isinstance(dict_space, dict) and key in dict_space:
                key_space = dict_space[key]

            # Apply noise based on value type
            if isinstance(value, np.ndarray):
                noisy_obs[key] = self._apply_to_array(value, key_space)
            elif isinstance(value, (int, float, np.integer, np.floating)):
                noisy_obs[key] = self._apply_to_scalar(value, key_space)
            else:
                # For unsupported types, keep unchanged
                noisy_obs[key] = value
        return noisy_obs

    def _apply_to_scalar(self,
                         scalar: Union[int, float, np.integer, np.floating],
                         space: Optional[spaces.Space] = None) -> Union[int, float]:
        """
        Apply Gaussian noise to a scalar observation.
        :param scalar: Scalar observation
        :param space: Space defining valid range for the scalar
        :return:  with added Gaussian noise
        """
        # Generate a single noise
        noise = self.rng.normal(loc=self.mean, scale=self.std)

        # Add noise
        noisy_scalar = scalar + noise

        # Handle different space types
        if space is not None:
            if isinstance(space, spaces.Discrete):
                # For Discrete spaces, round to nearest integer and clip to valid range
                noisy_scalar = int(round(np.clip(noisy_scalar, 0, space.n - 1)))
            elif isinstance(space, spaces.MultiDiscrete):
                # For MultiDiscrete, round and clip to valid range
                noisy_scalar = int(round(np.clip(noisy_scalar, 0, space.nvec - 1)))
            elif isinstance(space, spaces.Box):
                # For Box spaces, clip to bounds
                noisy_scalar = np.clip(noisy_scalar, space.low[0] if hasattr(space.low, '__len__') else space.low,
                                       space.high[0] if hasattr(space.high, '__len__') else space.high)

            # Preserve original type
        if isinstance(scalar, (int, np.integer)) and space is None:
            return int(round(noisy_scalar))

        return noisy_scalar
