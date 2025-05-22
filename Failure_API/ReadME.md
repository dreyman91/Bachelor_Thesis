
# failure-api

A modular failure injection API for Multi-Agent Reinforcement Learning (MARL) environments.  
Built on top of [PettingZoo](https://pettingzoo.farama.org/), this package introduces wrappers that simulate communication failures and observation noise between agents — enabling robust training under partial observability and degraded communication.



## Installation

```bash
pip install failure-api
````

---

##  Components

### Wrappers

| Wrapper                | Purpose                                                                  |
| ---------------------- | ------------------------------------------------------------------------ |
| `BaseWrapper`          | Provides utility functions such as RNG seeding.                          |
| `SharedObsWrapper`     | Enables shared observations across agents; used internally.              |
| `CommunicationWrapper` | Dynamically masks agent observations/actions based on failures.          |
| `NoiseWrapper`         | Injects noise into shared observations, excluding already masked values. |

---

### 🔌 Communication Models

| Model                | Description                                                                |
| -------------------- | -------------------------------------------------------------------------- |
| `ProbabilisticModel` | Drops communication links using Bernoulli sampling with fixed probability. |
| `BaseMarkovModel`    | Models temporally correlated failures with a Markov chain.                 |
| `DistanceModel`      | Drops communication based on exceeding a distance threshold.               |
| `DelayBasedModel`    | Delays messages with probabilistic delivery and expiration.                |
| `SignalBasedModel`   | Applies degradation using inverse-square signal loss and random drops.     |

---

### 🎙️ Noise Models

| Model                 | Description                                                          |
| --------------------- | -------------------------------------------------------------------- |
| `GaussianNoiseModel`  | Adds Gaussian noise: `X + N(μ, σ²)`; useful for random sensor noise. |
| `LaplacianNoiseModel` | Adds Laplacian noise: `X + Laplace(μ, b)`; good for sparse spikes.   |
| `CustomNoise`         | Accepts any callable function for custom distortion logic.           |

All models inherit from the `NoiseModel` base class and implement `.apply()`.

---

##  Example Usage

```python
from pettingzoo.mpe import simple_spread_v3
from failure_api.wrappers import CommunicationWrapper, NoiseWrapper
from failure_api.communication_models import ProbabilisticModel
from failure_api.noise_models import GaussianNoiseModel
from pettingzoo.utils import aec_to_parallel

# Base PettingZoo environment
env = simple_spread_v3.env(N=3, max_cycles=25)
agent_ids = env.possible_agents

# Apply communication failure model
model = ProbabilisticModel(agent_ids, failure_prob=0.3)
env = CommunicationWrapper(env, failure_models=[model])

# Add Gaussian noise to visible observations
env = NoiseWrapper(env, GaussianNoiseModel(std=0.5))

# Convert to parallel API
env = aec_to_parallel(env)

# Run one episode
observations = env.reset(seed=42)
for _ in range(10):
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    observations, rewards, terminations, truncations, infos = env.step(actions)
    print("Comm matrix:\n", env.get_communication_state())
```

---

## Custom Noise Example

```python
from failure_api.noise_models import CustomNoise

def zero_every_other(obs, space=None):
    obs[::2] = 0
    return obs

noise = CustomNoise(noise_fn=zero_every_other)
env = NoiseWrapper(env, noise_model=noise)
```

---

##  Features

* ✅ Pluggable communication failure and noise models
* 🔁 Compatible with both AEC and Parallel API in PettingZoo
* 🔒 Observation masking via active communication matrix
* 🎯 Observation noise excludes zero-masked (unseen) values
* 🤖 Supports MARL training frameworks (e.g. IQL, QMIX)
* 🧩 Fully extensible for custom failure scenarios

---

##  API Highlights

### `CommunicationWrapper`

* Applies observation/action masking using `ActiveCommunication`
* Works with any PettingZoo AEC environment
* Internally uses `SharedObsWrapper`
* Supports multi-model failure injection

### `NoiseWrapper`

* Adds noise only to visible observations
* Maintains shape and structure of observation space
* Customizable with user-defined noise logic

---

##  Requirements

* Python 3.8+
* `pettingzoo`
* `gymnasium`
* `numpy`
* `scipy`
* *(Optional)*: `matplotlib`, `tqdm`, `networkx`

---

##  Testing

Use this to inspect the communication mask during runtime:

```python
print(env.get_communication_state())  # Visualize active communication links
```

---

##  Extendability

You can define your own failure or noise model by subclassing:

```python
from failure_api.communication_models import CommunicationModels

class MyFailureModel(CommunicationModels):
    def update_connectivity(self, comm_matrix):
        # Custom logic here
        return comm_matrix
```

---

##  Citation

If you use this in academic work, please cite:

```bibtex
@bachelorsthesis{adegun2025marlapi,
  author    = {Oluwadamilare Israel Adegun},
  title     = {Development of a Multi-Agent Reinforcement Learning API for Dynamic Observation and Action Spaces},
  school    = {University of Duisburg-Essen},
  year      = {2025},
  note      = {[Online; accessed 21-May-2025]}
}
```

---

## 🔗 Links

* 🔗 GitHub: [dreyman91/Bachelor\_Thesis](https://github.com/dreyman91/Bachelor_Thesis)
* 📦 PyPI: [pypi.org/project/failure-api](https://pypi.org/project/failure-api/)

---



