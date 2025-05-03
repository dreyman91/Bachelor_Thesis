import pytest
import numpy as np
import time
from gymnasium import spaces

from Failure_API.src.failure_api.noise_models.gaussian_noise import GaussianNoiseModel


# ----- Unit Tests -----

def test_initialization():
    """Test that the model initializes correctly with valid parameters."""
    mean = 0.0
    std = 0.5

    model = GaussianNoiseModel(mean=mean, std=std)

    assert model.mean == mean, "Mean was not stored correctly"
    assert model.std == std, "Standard deviation was not stored correctly"
    assert model.rng is None, "RNG should be None until set"

def test_initialization_invalid_params():
    """Test that the model handles invalid parameters appropriately."""
    # Currently the GaussianNoiseModel doesn't validate parameters
    # If validation is added, tests should be added here
    pass


def test_set_rng(fixed_rng):
    """Test that the RNG can be set correctly."""
    model = GaussianNoiseModel(mean=0.0, std=1.0)

    # RNG should be None initially
    assert model.rng is None, "RNG should be None before setting"

    # Set RNG
    model.set_rng(fixed_rng)

    # RNG should be set now
    assert model.rng is fixed_rng, "RNG was not set correctly"


def test_apply_to_array(fixed_rng):
    """Test applying noise to a numpy array."""
    mean = 0.0
    std = 1.0

    model = GaussianNoiseModel(mean=mean, std=std)
    model.set_rng(fixed_rng)

    # Create a test array of zeros
    test_array = np.zeros(10, dtype=float)

    # Apply noise
    noisy_array = model.apply(test_array)

    # Check that noise was added
    assert not np.array_equal(noisy_array, test_array), "No noise was added to the array"

    # Check that the arrays have the same shape
    assert noisy_array.shape == test_array.shape, "Noisy array shape doesn't match original"

    assert -3 * std <= np.mean(noisy_array) <= 3 * std, "Mean outside expected range"
    assert 0.5 * std <= np.std(noisy_array) <= 1.5 * std, "Standard deviation outside expected range"


    # Test with observation_space
    box_space = spaces.Box(low=-2.0, high=2.0, shape=(10,))

    # Create array with values that will exceed the bounds when noise is added
    extreme_array = np.ones(10) * 1.5  # Close to upper bound

    # Apply noise with observation space
    noisy_extreme = model.apply(extreme_array, box_space)

    # Check that values are clipped to observation space
    assert np.all(noisy_extreme >= -2.0), "Values below lower bound after noise application"
    assert np.all(noisy_extreme <= 2.0), "Values above upper bound after noise application"


def test_apply_to_dict(fixed_rng):
    """Test applying noise to a dictionary observation."""
    mean = 0.0
    std = 1.5

    model = GaussianNoiseModel(mean=mean, std=std)
    model.set_rng(fixed_rng)

    # Create a test dictionary observation
    test_dict = {
        "position": np.array([1.0, 2.0, 3.0]),
        "velocity": np.array([0.1, 0.2, 0.3]),
        "scalar": 5.0,
        "integer": 10,
        "string": "not_noisy"  # Should be left unchanged
    }

    # Apply noise
    noisy_dict = model.apply(test_dict)

    # Check that numeric values were modified
    assert not np.array_equal(noisy_dict["position"], test_dict["position"]), \
        "No noise added to position array"
    assert not np.array_equal(noisy_dict["velocity"], test_dict["velocity"]), \
        "No noise added to velocity array"
    assert noisy_dict["scalar"] != test_dict["scalar"], \
        "No noise added to scalar value"
    assert noisy_dict["integer"] != test_dict["integer"], \
        "No noise added to integer value"

    # Check that non-numeric values were left unchanged
    assert noisy_dict["string"] == test_dict["string"], \
        "String value should not be modified"

    # Test with observation space
    dict_space = spaces.Dict({
        "position": spaces.Box(low=-5.0, high=5.0, shape=(3,)),
        "velocity": spaces.Box(low=-1.0, high=1.0, shape=(3,)),
        "scalar": spaces.Box(low=0.0, high=10.0, shape=(1,)),
        "integer": spaces.Discrete(20)
    })

    # Apply noise with observation space
    noisy_dict_bounded = model.apply(test_dict, dict_space)

    # Check that values are clipped to observation spaces
    assert np.all(noisy_dict_bounded["position"] >= -5.0), "Position below lower bound"
    assert np.all(noisy_dict_bounded["position"] <= 5.0), "Position above upper bound"
    assert np.all(noisy_dict_bounded["velocity"] >= -1.0), "Velocity below lower bound"
    assert np.all(noisy_dict_bounded["velocity"] <= 1.0), "Velocity above upper bound"
    assert noisy_dict_bounded["scalar"] >= 0.0, "Scalar below lower bound"
    assert noisy_dict_bounded["scalar"] <= 10.0, "Scalar above upper bound"
    assert 0 <= noisy_dict_bounded["integer"] < 20, "Integer outside valid range"

def test_apply_to_scalar(fixed_rng):
    """Test applying noise to scalar values."""
    mean = 0.0
    std = 1.5

    model = GaussianNoiseModel(mean=mean, std=std)
    model.set_rng(fixed_rng)

    # Test with integer
    test_int = 10
    noisy_int = model.apply(test_int)

    # Result should be a float (or an int if rounded)
    assert isinstance(noisy_int, (int, float)), "Result should be numeric"
    assert noisy_int != test_int, "No noise added to integer"

    # Test with float
    test_float = 5.0
    noisy_float = model.apply(test_float)

    assert isinstance(noisy_float, float), "Result should be a float"
    assert noisy_float != test_float, "No noise added to float"

    # Test with observation space
    box_space = spaces.Box(low=0.0, high=10.0, shape=(1,))

    # Apply noise with observation space
    noisy_float_bounded = model.apply(test_float, box_space)

    # Check that value is clipped to observation space
    assert noisy_float_bounded >= 0.0, "Value below lower bound after noise application"
    assert noisy_float_bounded <= 10.0, "Value above upper bound after noise application"

def test_apply_to_unsupported_type(fixed_rng):
    """Test applying noise to unsupported types."""
    model = GaussianNoiseModel(mean=0.0, std=1.0)
    model.set_rng(fixed_rng)

    # Test with string (should return unchanged)
    test_string = "hello"
    result = model.apply(test_string)
    assert result == test_string, "String should be returned unchanged"

    # Test with None (should return unchanged)
    result = model.apply(None)
    assert result is None, "None should be returned unchanged"

    # Test with list (should return unchanged since it's not a numpy array)
    test_list = [1, 2, 3]
    result = model.apply(test_list)
    assert result == test_list, "List should be returned unchanged"


def test_noise_statistical_properties(fixed_rng):
    """Test that the generated noise has the expected statistical properties."""
    mean = 0.5
    std = 2.0

    model = GaussianNoiseModel(mean=mean, std=std)
    model.set_rng(fixed_rng)

    # Generate a large number of noise samples
    original = np.zeros(10000)
    noisy = model.apply(original)
    noise = noisy - original

    # Check statistical properties
    assert np.isclose(np.mean(noise), mean, atol=0.1), \
        f"Noise mean {np.mean(noise)} doesn't match expected {mean} within tolerance"
    assert np.isclose(np.std(noise), std, atol=0.1), \
        f"Noise std {np.std(noise)} doesn't match expected {std} within tolerance"

    # Check that noise follows a normal distribution (using histogram)
    hist, bin_edges = np.histogram(noise, bins=20, density=True)

    # Calculate expected normal distribution
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    expected_pdf = 1 / (std * np.sqrt(2 * np.pi)) * np.exp(-((bin_centers - mean) ** 2) / (2 * std ** 2))

    # Rough check that histogram shape matches expected PDF
    # This is a bit imprecise but checks the general shape
    correlation = np.corrcoef(hist, expected_pdf)[0, 1]
    assert correlation > 0.9, \
        f"Noise distribution doesn't match expected normal distribution, correlation: {correlation}"


def test_rng_error_handling():
    """Test that the model raises an error if RNG is not set before use."""
    model = GaussianNoiseModel(mean=0.0, std=1.0)

    # Applying noise without setting RNG should raise an error
    with pytest.raises(ValueError, match="Random number generator not initialized."):
        model.apply(np.zeros(10))


def test_missing_space_items(fixed_rng):
    """Test handling of missing items in observation space."""
    model = GaussianNoiseModel(mean=0.0, std=1.0)
    model.set_rng(fixed_rng)

    # Create dict observation with more items than in space
    test_dict = {
        "position": np.array([1.0, 2.0, 3.0]),
        "velocity": np.array([0.1, 0.2, 0.3]),
        "extra": 5.0  # Not in observation space
    }

    # Create observation space without all items
    dict_space = spaces.Dict({
        "position": spaces.Box(low=-5.0, high=5.0, shape=(3,)),
        "velocity": spaces.Box(low=-1.0, high=1.0, shape=(3,))
        # No "extra" item
    })

    # Apply noise
    noisy_dict = model.apply(test_dict, dict_space)

    # Check that all items were handled correctly
    assert "position" in noisy_dict, "Position should be in result"
    assert "velocity" in noisy_dict, "Velocity should be in result"
    assert "extra" in noisy_dict, "Extra item should still be in result"

    # Position and velocity should be clipped to space
    assert np.all(noisy_dict["position"] >= -5.0), "Position below lower bound"
    assert np.all(noisy_dict["position"] <= 5.0), "Position above upper bound"
    assert np.all(noisy_dict["velocity"] >= -1.0), "Velocity below lower bound"
    assert np.all(noisy_dict["velocity"] <= 1.0), "Velocity above upper bound"

    # Extra item should have noise but no clipping
    assert noisy_dict["extra"] != test_dict["extra"], "No noise added to extra item"


# ----- Performance Tests -----

def test_performance_large_array(fixed_rng):
    """Test performance with a large array."""
    model = GaussianNoiseModel(mean=0.0, std=1.0)
    model.set_rng(fixed_rng)

    # Create a large array
    large_array = np.zeros((1000, 1000))

    # Measure performance
    start_time = time.time()
    model.apply(large_array)
    execution_time = time.time() - start_time

    # Should be able to process a 1000x1000 array in under 0.1 seconds
    assert execution_time < 0.1, \
        f"Performance test failed: {execution_time:.3f} seconds for 1000x1000 array (threshold: 0.1s)"


def test_performance_large_dict(fixed_rng):
    """Test performance with a large dictionary observation."""
    model = GaussianNoiseModel(mean=0.0, std=1.0)
    model.set_rng(fixed_rng)

    # Create a large dictionary observation
    large_dict = {}
    for i in range(100):
        large_dict[f"array_{i}"] = np.random.rand(100)

    # Measure performance
    start_time = time.time()
    model.apply(large_dict)
    execution_time = time.time() - start_time

    # Should be able to process a dictionary with 100 arrays in under 0.05 seconds
    assert execution_time < 0.05, \
        f"Performance test failed: {execution_time:.3f} seconds for large dict (threshold: 0.05s)"


# ----- PettingZoo Compatibility Tests -----

def test_pettingzoo_compatibility(pettingzoo_env, fixed_rng):
    """Test that the model can be integrated with PettingZoo environments."""
    model = GaussianNoiseModel(mean=0.0, std=0.5)
    model.set_rng(fixed_rng)

    # Reset the environment to get initial observations
    observations, infos = pettingzoo_env.reset(seed=42)

    # Apply noise to observations
    noisy_observations = {}
    for agent, obs in observations.items():
        # If observation is an array-like structure
        if isinstance(obs, (np.ndarray, dict)):
            # Get the observation space for this agent
            obs_space = pettingzoo_env.observation_space(agent)

            # Apply noise with observation space
            noisy_obs = model.apply(obs, obs_space)

            # Verify noise was added
            if isinstance(obs, np.ndarray):
                assert not np.array_equal(noisy_obs, obs), \
                    f"No noise was added to {agent}'s observation"

            noisy_observations[agent] = noisy_obs
        else:
            # For non-supported types, just pass through
            noisy_observations[agent] = obs

    # Generate random actions based on noisy observations
    actions = {}
    for agent in pettingzoo_env.agents:
        # In a real implementation, the agent would use noisy_observations[agent] to decide
        # For testing, we just sample a random action
        actions[agent] = pettingzoo_env.action_space(agent).sample()

    # Step the environment with the actions
    pettingzoo_env.step(actions)