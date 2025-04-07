import numpy as np
from gymnasium import spaces

from Failure_API.src.wrapper_api.models.noise_model import GaussianNoise

def test_gaussian_noise_variation_and_clipping():
    model = GaussianNoise(mean=0.0, std=0.2)

    obs = np.array([0.6, 0.5], dtype=np.float32)
    obs_space = spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)
    noisy_obs = model.apply(obs, obs_space)

    assert noisy_obs.shape == obs.shape
    assert np.all(noisy_obs >= 0.0), "Gaussian Noise dropped below  min bound"
    assert np.all(noisy_obs <= 1.0), "Gaussian Noise exceeded max bound"
    assert not np.allclose(obs, noisy_obs), "Gaussian Noise had no effect"