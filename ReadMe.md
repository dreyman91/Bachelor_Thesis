
This project provides a modular failure injection API for Multi-Agent Reinforcement Learning (MARL) environments. Built on top of [PettingZoo](https://pettingzoo.farama.org/), the API introduces wrappers that dynamically simulate communication failures and observation noise between agents. It supports realistic training under partial observability, degraded communication, and noisy environments.

---

##  Components

### Wrappers

| Wrapper | Purpose |
|--------|---------|
| `BaseWrapper` | Common utilities such as RNG seeding. |
| `SharedObsWrapper` | Enables shared observations for all agents. Required for communication masking. |
| `CommunicationWrapper` | Dynamically masks agent observations and actions based on communication failures. |
| `NoiseWrapper` | Adds noise to shared observations (except zeros) to simulate signal degradation. |

### 🔌 Communication Models

| Model | Description |
|-------|-------------|
| `ProbabilisticModel` | Bernoulli-based dropout of links with fixed probability. |
| `BaseMarkovModel` | Markov chain per link for temporal correlation in communication failures. |
| `DistanceModel` | Bandwidth degrades or drops if agent distance exceeds a threshold. |
| `DelayBasedModel` | Queues messages with probabilistic delay and expiration. |
| `SignalBasedModel` | Simulates packet loss and degradation based on inverse-square signal law. |

---

## 🎙️ Noise Models

The noise models simulate uncertainty and distortion in agent observations. They are designed to work with `NoiseWrapper` and apply noise selectively to shared observations without affecting masked (zero) values.

### Available Noise Models


| Model                | Description                                                                                  |
|---------------------|----------------------------------------------------------------------------------------------|
| `GaussianNoiseModel` | Adds Gaussian noise: `X_noisy = X + N(μ, σ²)`. Suitable for sensor-like random fluctuations. |
| `LaplacianNoiseModel` | Adds Laplacian noise: `X_noisy = X + Laplace(μ, b)`. Models occasional large spikes and sensor outliers. |
| `CustomNoise`        | Accepts any user-defined callable function for noise injection. Allows custom distortion logic per observation. |
All models inherit from the abstract `NoiseModel` base class, which enforces a `.apply()` interface and supports random seeding.

---

## 💡 Example Usage

```python
# SharedOBS wrapper is already instantiated in Communication wrapper
import numpy as np
import pandas as pd
from mpe2 import simple_spread_v3
from failure_api.wrappers import (CommunicationWrapper,
                                  NoiseWrapper)
from failure_api.noise_models import (GaussianNoiseModel)
from failure_api.communication_models import ProbabilisticModel, ActiveCommunication
from pettingzoo.utils import aec_to_parallel


# Step 1: Create base environment
env = simple_spread_v3.env(N=3, max_cycles=25)
agent_ids = env.possible_agents

# Step 2: Apply communication wrapper with probabilistic failures
# Each communication link has a 30% chance of failing
model = ProbabilisticModel(agent_ids, failure_prob=0.3)
wrapped_comm_env = CommunicationWrapper(env, failure_models=[model])

# Step 3: Apply noise to successfully transmitted observations
wrapped_ns_env = NoiseWrapper(wrapped_comm_env, GaussianNoiseModel(std=0.5))

# Step 4: Convert wrapper from aec mode to parallel mode
cv_wrapped = aec_to_parallel(wrapped_ns_env)

# Step 5: Run the environment 10 times
observations = cv_wrapped.reset(seed=42)
for _ in range(10):
    # Sample random actions
    actions = {agent: cv_wrapped.action_space(agent).sample() for agent in cv_wrapped.agents}

    # Step the environment
    observations, rewards, terminations, truncations, infos = cv_wrapped.step(actions)

    # Check agents that can communicate with each other
    comms = wrapped_ns_env.get_communication_state().astype(int)
    print(f"Step{_}, Number of active communication links: {np.sum(comms)}")

    # Check if episode is done
    if all(terminations.values()) or all(truncations.values()):
        break

print("Successful")
```

To add observation noise:

```python
from noise_wrapper import NoiseWrapper
from gaussian_noise import GaussianNoiseModel

noise_model = GaussianNoiseModel(mean=0.0, std=0.05)
env = NoiseWrapper(env, noise_model=noise_model)
```

Custom noise example:

```python
from custom_noise import CustomNoise

def zero_every_other(obs, space=None):
    obs[::2] = 0
    return obs

noise_model = CustomNoise(noise_fn=zero_every_other)
env = NoiseWrapper(env, noise_model=noise_model)
```

---

## Features

-  Seamless integration with PettingZoo AEC environments  
-  Pluggable failure models for communication dropout, delay, or signal loss  
-  Optional noise injection on masked observations  
-  Modular design for composing multiple wrappers  
-  Compatible with MARL training (e.g., IQL, QMIX)  

---

##  Documentation

### `CommunicationWrapper`
- Applies masking logic using `ActiveCommunication` matrix
- Uses `SharedObsWrapper` to simulate shared visibility
- Supports multiple simultaneous failure models
- Automatically replaces actions with no-op if agent is isolated

### `NoiseWrapper`
- Injects noise into shared observations, excluding already masked (zero) values
- Supports pluggable noise models (Gaussian, Laplacian, adversarial, etc.)
- Preserves observation shape and semantics

---

##  Requirements

- Python 3.8+
- `pettingzoo`
- `numpy`
- `gymnasium`
- Optional: `matplotlib`, `scipy`, `tqdm`, `networkx`

---

##  Testing

To verify masking or noise behavior, enable debugging or check communication states:

```python
print(env.get_communication_state())  # Visualize connectivity
```

---

## Extendability

Create your own failure or noise models by subclassing `CommunicationModels` or `NoiseModel`, and plug them into the wrappers. Example:

```python
class MyFailureModel(CommunicationModels):
    def update_connectivity(self, comms_matrix):
        # custom logic
        pass
```

---

##  Citation

If used in academic work, cite as:

```bibtex
@bachelorsthesis{adegun2025marlapi,
  author    = {Oluwadamilare Israel Adegun},
  title     = {Development of a Multi-Agent Reinforcement Learning API for Dynamic Observation and Action Spaces},
  school    = {University of Duisburg-Essen},
  year      = {2025},
  note      = {[Online; accessed 21-May-2025]}
}
```