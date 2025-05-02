import numpy as np
from typing import Optional, Any, Dict, Union
from gymnasium import spaces
from ..noise_models.base_noise_model import NoiseModel
import logging

logger = logging.getLogger(__name__)


class LaplacianNoiseModel(NoiseModel):
    """
    Adds Laplacian (heavy-tailed) noise to observations.

    This model implements the addition of noise sampled from a Laplace distribution:

    X_noisy = X + Laplace(μ, b)

    Where:
    - X is the original observation
    - μ is the location parameter (typically 0)
    - b is the scale parameter controlling the "spread" of the distribution
    - Laplace(μ, b) represents a random sample from a Laplace distribution

    Compared to Gaussian noise, Laplacian noise has heavier tails, meaning it's
    more likely to produce occasional large deviations. This makes it useful for
    modeling sensor errors that sometimes have significant outliers.
    """

    def __init__(self, loc: float = 0.0, scale: float = 0.1):
        super().__init__()
        self.loc = loc
        self.scale = scale

    def apply(self, obs: Any, observation_space: Optional[spaces.Space] = None) -> Any:
        """
        Apply Laplacian noise to an observation.
        :param obs: Observation to which noise will be added
        :param observation_space: Space defining the valid range for the observation (optional)
        :return:  Observation with added Laplacian noise
        """
        if self.rng is None:
            logger.warning("Random number generator not initialized. call set_rng() first.")
        if isinstance(obs, np.ndarray):
            self._apply_to_array(obs, observation_space)
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
        Apply Laplacian noise to a numpy array observation.
        :param array: Array observation
        :param space: Space defining valid range for the array
        :return:  with added Laplacian noise
        """
        # Generate Laplacian noise of the same shape as the input array
        noise = self.rng.laplace(loc=self.loc, scale=self.scale, size=array.shape)

        # Add noise to observation
        noisy_array = array + noise

        # Clip to space bounds if provided
        if space is not None and isinstance(space, spaces.Box):
            noisy_array = np.clip(noisy_array, space.low, space.high)

        return noisy_array

    def _apply_to_dict(self,
                       dict_obs: Dict,
                       dict_space: Optional[Dict[str, spaces.Space]] = None) -> Dict:
        """
        Apply Laplacian noise to a dictionary observation.
        :param dict_obs: Dictionary observation
        :param dict_space: Dictionary of spaces defining valid ranges
        :return: Dictionary with added Laplacian noise
        """
        noisy_obs = {}

        for key, value in dict_obs.items():
            # Get space for this key if available
            key_space = None
            if dict_space is not None and key in dict_space:
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
        Apply Laplacian noise to a scalar observation.
        :param scalar: Scalar observation
        :param space: Space defining valid range for the scalar
        :return: Scalar with added Gaussian noise
        """
        # Generate a single Laplacian noise value
        noise = self.rng.laplace(loc=self.loc, scale=self.scale)

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
                noisy_scalar = np.clip(noisy_scalar,
                                       space.low[0] if hasattr(space.low, '__len__') else space.low,
                                       space.high[0] if hasattr(space.high, '__len__') else space.high)

        # Preserve original type
        if isinstance(scalar, (int, np.integer)) and space is None:
            return int(round(noisy_scalar))

        return noisy_scalar
