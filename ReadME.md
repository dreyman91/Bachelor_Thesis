
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

###  Communication Models

| Model                | Description                                                                |
| -------------------- | -------------------------------------------------------------------------- |
| `ProbabilisticModel` | Drops communication links using Bernoulli sampling with fixed probability. |
| `BaseMarkovModel`    | Models temporally correlated failures with a Markov chain.                 |
| `DistanceModel`      | Drops communication based on exceeding a distance threshold.               |
| `DelayBasedModel`    | Delays messages with probabilistic delivery and expiration.                |
| `SignalBasedModel`   | Applies degradation using inverse-square signal loss and random drops.     |

---

###  Noise Models

| Model                 | Description                                                          |
| --------------------- |----------------------------------------------------------------------|
| `GaussianNoiseModel`  | Adds Gaussian noise: `X + N(μ, σ²)`; useful for random sensor noise. |
| `LaplacianNoiseModel` | Adds Laplacian noise: `X + Laplace(μ, b)`; good for sparse spikes.   |
| `CustomNoise`         | Accepts any callable function for custom distortion logic.           |

All models inherit from the `NoiseModel` base class and implement `.apply()`.

---

##  Example Usage

```python
from mpe2 import simple_spread_v3
from failure_api.wrappers import CommunicationWrapper, NoiseWrapper
from failure_api.communication_models import ProbabilisticModel, BaseMarkovModel
from failure_api.noise_models import GaussianNoiseModel
from pettingzoo.utils import aec_to_parallel

# Base PettingZoo environment
env = simple_spread_v3.env(N=3, max_cycles=25)
agent_ids = env.possible_agents

# Apply communication failure model
model = ProbabilisticModel(agent_ids=agent_ids, failure_prob=0.8)
wrapped_env = CommunicationWrapper(env, failure_models=[model])

# Convert to parallel API
parallel_env = aec_to_parallel(wrapped_env)

# Run Simulation
observations, _  = parallel_env.reset(seed=42)
initial_comm_matrix = wrapped_env.get_communication_state().astype(int)
print(f"\nInitial Communication Matrix (0=masked, 1=visible)\n")
print(initial_comm_matrix)

for _ in range(10):
    actions = {agent: parallel_env.action_space(agent).sample() for agent in parallel_env.agents}
    observations, rewards, terminations, truncations, infos = parallel_env.step(actions)
    print(f"Masked State: (0=masked, 1=visible)\n", wrapped_env.get_communication_state())
    
    if all(terminations.values()) or all(truncations.values()):
        break

    
#%%
# OPTIONAL: Noise model, can be used with/without Communication wrapper
#  to compare observations before noise is injected, the communication wrapper
# needs to be called to get the raw observation.


# Base PettingZoo environment
noise_env = simple_spread_v3.env(N=3, max_cycles=25)
agent_ids = env.possible_agents

comm_wrapper = CommunicationWrapper(noise_env, failure_models=[ProbabilisticModel(
    agent_ids, failure_prob=0.8
)]) # step will not be taken so as not to mask observations 

# convert to parallel env
parallel_comm_wrapper = aec_to_parallel(comm_wrapper)
comm_obs, _ = parallel_comm_wrapper.reset(seed=42)
comm_obs_agent_0 = comm_obs[agent_ids[0]]

# Inject Noise 
noise_wrapper = NoiseWrapper(comm_wrapper, noise_model=
                             GaussianNoiseModel(mean=0.1, std=0.2 ))

# Convert to parallel API
noise_parallel_env = aec_to_parallel(noise_wrapper)

# Run Simulation
noisy_obs, _  = noise_parallel_env.reset(seed=42)

noisy_obs_agent_0 = noisy_obs[agent_ids[0]]

print("\nComparison for agent_0")
for other_agent in comm_obs_agent_0:
    print(f"\nObservation from {other_agent}:")
    print(f" Clean:")
    print(f"{comm_obs_agent0[other_agent]:.3f}")
    
    print(f"\n Noisy :")
    print(f"{noisy_obs_agent0[other_agent]:.3f}")
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



