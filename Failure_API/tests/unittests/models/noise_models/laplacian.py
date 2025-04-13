import numpy as np
from gymnasium import spaces

from Failure_API.src.wrapper_api.noise_models.base_noise_model import LaplacianNoise

def test_gaussian_noise_variation_and_clipping():
    model = LaplacianNoise(scale=0.1, loc=0.4)

    obs = np.array([0.4, 0.8], dtype=np.float32)
    obs_space = spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)
    noisy_obs = model.apply(obs, obs_space)

    assert noisy_obs.shape == obs.shape
    assert np.all(noisy_obs >= 0.0), "Laplacian Noise dropped below  min bound"
    assert np.all(noisy_obs <= 1.0), "Laplacian Noise exceeded max bound"
    assert not np.allclose(obs, noisy_obs), "Laplacian Noise had no effect"