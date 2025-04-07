import numpy as np
from gymnasium import spaces

from Failure_API.src.wrapper_api.models.noise_model import CustomNoise
def test_custom_noise():
    def noise_fn(obs, space):
        return np.clip(obs + 0.3, space.low, space.high)

    model = CustomNoise(noise_fn=noise_fn)

    obs = np.array([0.6, 0.5], dtype=np.float32)

    obs_space = spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)

    noisy_obs = model.apply(obs, obs_space)

    assert noisy_obs.shape == obs.shape
    assert np.all(noisy_obs <= 1.0), "Values exceeds upper bound"
    assert np.all(noisy_obs >= 0.0), "Values exceeds lower bound"
    assert not np.allclose(obs, noisy_obs), "No noise was applied"