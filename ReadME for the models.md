# Delay-Based Communication Model

## Overview

The Delay-Based Communication Model simulates message propagation with variable delays between agents. 
This model is particularly useful for researchers studying distributed algorithms, multi-agent systems, 
or network protocols where communication delays significantly impact system behavior.

## Mathematical Foundation

The model is based on a queue-based approach to message propagation:

1. When an agent sends a message, the message is assigned a delay value (d)
2. The message is placed in a queue for d time steps
3. When d reaches zero, the message is delivered to the recipient
4. With probability p, a message may be corrupted/dropped during transmission
5. New messages are generated based on the current connectivity state

## Core Model

The `BaseDelayModel` provides a pure implementation of this mathematical model, with:

- Uniform random sampling of delays between min_delay and max_delay
- Configurable message drop probability
- Efficient queue-based implementation
- Clear separation from scenario-specific extensions

## Using the Base Model

Researchers can use the base model directly for experiments that require a 
simple, well-defined delay model:

```python
from failure_api.communication_models import BaseDelayModel
from failure_api.active_communication import ActiveCommunication

# Create a model with delays between 1-5 time steps
model = BaseDelayModel(
    agent_ids=["agent1", "agent2", "agent3"], 
    min_delay=1, 
    max_delay=5
)

# Initialize communication matrix
comms_matrix = ActiveCommunication(model.agent_ids)

# Update connectivity (typically called in simulation loop)
model.update_connectivity(comms_matrix)
```



## Markov Chain Communication Model

## Overview

The Markov Chain Communication Model simulates communication reliability between agents 
using a stochastic approach based on Markov chains. This model is particularly useful 
for researchers studying distributed systems, robotic swarms, sensor networks, or any 
multi-agent system where communication reliability is imperfect and probabilistic.

## Mathematical Foundation

The model represents each communication link as a 2-state Markov chain:
- State 0: Disconnected (no communication possible)
- State 1: Connected (communication possible)

Each link has a 2×2 transition probability matrix P where:
- P[0,0]: Probability of staying disconnected
- P[0,1]: Probability of changing from disconnected to connected
- P[1,0]: Probability of changing from connected to disconnected
- P[1,1]: Probability of staying connected

At each time step, the state of each link transitions according to these probabilities.

## Core Model

The `BaseMarkovModel` provides a pure implementation of this mathematical model, with:

- Link-specific transition matrices
- Default matrices when specific ones aren't provided
- Proper validation and normalization of transition probabilities
- Efficient state tracking

## Using the Base Model

Researchers can use the base model directly for experiments requiring 
a mathematically pure Markov chain approach:

```python
from failure_api.communication_models import BaseMarkovModel
from failure_api.active_communication import ActiveCommunication
import numpy as np

# Define transition matrices (optional)
# Higher reliability link: 95% chance of staying connected
high_reliability = np.array([
    [0.8, 0.2],  # 20% chance to recover when disconnected
    [0.05, 0.95]  # Only 5% chance to disconnect when connected
])

# Lower reliability link: 70% chance of staying connected
low_reliability = np.array([
    [0.9, 0.1],  # Only 10% chance to recover when disconnected
    [0.3, 0.7]   # 30% chance to disconnect when connected
])

# Create transition probability dictionary
transitions = {
    ("agent1", "agent2"): high_reliability,
    ("agent2", "agent1"): high_reliability,
    ("agent2", "agent3"): low_reliability,
    ("agent3", "agent2"): low_reliability
}

# Create model
model = BaseMarkovModel(
    agent_ids=["agent1", "agent2", "agent3"],
    transition_probabilities=transitions
)

# Initialize communication matrix
comms_matrix = ActiveCommunication(model.agent_ids)

# Update connectivity (typically called in simulation loop)
model.update_connectivity(comms_matrix)
```



# Adversarial Jamming Model

## Overview

The Adversarial Jamming Model simulates communication disruption in multi-agent systems,
where an adversary can selectively block or degrade communication between agents.
This model is particularly valuable for researchers studying communication security,
resilience against attacks, and robust multi-agent coordination in hostile environments.

## Mathematical Foundation

The model represents jamming as a binary decision for each communication link:
- Jammed: Communication is either blocked or degraded
- Not jammed: Communication proceeds normally

When jamming occurs, it can have two effects:
1. Complete blocking: Communication is entirely prevented
2. Signal degradation: Communication quality is reduced by adding noise

The jamming decision can be based on various factors, including:
- Temporal patterns (schedule-based jamming)
- Spatial zones (location-based jamming)
- Specific targets (targeted jamming)
- Persistent effects (state-dependent jamming)

## Core Model

The `BaseJammingModel` provides a pure implementation focused on the core jamming mechanics:

- Binary jamming decisions for each link
- Choice between complete blocking and signal degradation
- Extension points for different jamming strategies
- Tracking of jamming states over time

## Using the Base Model

Researchers can use the base model as a foundation for implementing custom jamming strategies:

```python
from failure_api.communication_models import BaseJammingModel
from failure_api.active_communication import ActiveCommunication

# Create a custom jamming strategy
def custom_jamming_strategy(sender, receiver, context=None):
    # Implement custom logic to determine if link should be jammed
    return (sender == "agent1" and receiver == "agent2")

# Create basic jamming model
class CustomJammingModel(BaseJammingModel):
    def is_jammed(self, sender, receiver, context=None):
        return custom_jamming_strategy(sender, receiver, context)

# Initialize model
model = CustomJammingModel(
    agent_ids=["agent1", "agent2", "agent3"],
    full_block=True  # Complete blocking when jammed
)

# Initialize communication matrix
comms_matrix = ActiveCommunication(model.agent_ids)

# Update connectivity (typically called in simulation loop)
model.update_connectivity(comms_matrix)
```


# Signal-Based Communication Model

## Overview

The Signal-Based Communication Model simulates realistic wireless signal propagation
between agents in a physical environment. This model is especially valuable for researchers
studying mobile robot teams, wireless sensor networks, UAV swarms, or any multi-agent
system where communication is constrained by physical distance and signal properties.

## Mathematical Foundation

The model is based on fundamental principles of electromagnetic wave propagation:

1. **Inverse-Square Law**:
   Signal strength diminishes with the square of the distance:
   
   S = P / (d² + ε)
   
   Where:
   - S is the received signal strength
   - P is the transmission power
   - d is the distance between transmitter and receiver
   - ε is a small constant to prevent division by zero

2. **Probabilistic Packet Loss**:
   Probability of successful packet transmission decreases exponentially with distance:
   
   P(success) = e^(-α·d)
   
   Where:
   - α is the dropout parameter (higher values cause faster decay)
   - d is the distance between transmitter and receiver

3. **Signal Threshold**:
   Communication is only possible when signal strength exceeds a minimum threshold:
   
   S ≥ S_min

4. **Spatial Efficiency**:
   The model uses KD-Tree spatial indexing for computational efficiency,
   allowing it to scale to large numbers of agents.

## Core Model

The `BaseSignalModel` provides a pure implementation of these physical principles, with:

- Physics-based signal propagation using the inverse-square law
- Distance-dependent probabilistic packet loss
- Configurable transmission power and signal thresholds
- Efficient spatial indexing for performance

## Using the Base Model

Researchers can use the base model directly for experiments requiring
realistic wireless communication dynamics:

```python
from failure_api.communication_models import BaseSignalModel
from failure_api.active_communication import ActiveCommunication
import numpy as np

# Define position function
def get_positions():
    # This would typically come from your simulation environment
    return {
        "agent1": np.array([0.0, 0.0]),
        "agent2": np.array([3.0, 4.0]),  # 5 units from agent1
        "agent3": np.array([10.0, 10.0])  # ~14.1 units from agent1
    }

# Create model with realistic parameters
model = BaseSignalModel(
    agent_ids=["agent1", "agent2", "agent3"],
    position_fn=get_positions,
    tx_power=15.0,  # Transmission power
    min_signal_strength=0.01,  # Minimum detectable signal
    dropout_alpha=0.2  # Packet loss parameter
)

# Initialize communication matrix
comms_matrix = ActiveCommunication(model.agent_ids)

# Update connectivity (typically called in simulation loop)
model.update_connectivity(comms_matrix)
```


# Probabilistic Communication Model

## Overview

The Probabilistic Communication Model implements a simple stochastic approach to
communication reliability, where each communication attempt has a fixed probability
of success or failure, independent of all other factors.

## Mathematical Foundation

This model is based on the Bernoulli distribution, one of the simplest probability
distributions in statistics:

P(X = k) = p^k * (1-p)^(1-k) for k ∈ {0, 1}

Where:
- p is the probability of success (communication succeeds)
- (1-p) is the probability of failure (communication fails)
- Each trial is independent of all other trials

This creates a memoryless stochastic process where:
- Past communication successes/failures do not affect future probabilities
- All agent pairs have the same probability of communication failure
- Each time step represents independent Bernoulli trials

## Core Model

The `BaseProbabilisticModel` provides a pure implementation of this mathematical model, with:

- Fixed failure probability for all communication links
- Independent trials at each time step
- Configurable bandwidth for successful communication

## Using the Base Model

Researchers can use this model for experiments requiring simple, uniform
communication reliability:

```python
from failure_api.communication_models import BaseProbabilisticModel
from failure_api.active_communication import ActiveCommunication

# Create model with 20% chance of communication failure
model = BaseProbabilisticModel(
    agent_ids=["agent1", "agent2", "agent3"],
    failure_probability=0.2,
    max_bandwidth=1.0
)

# Initialize communication matrix
comms_matrix = ActiveCommunication(model.agent_ids)

# Update connectivity (typically called in simulation loop)
model.update_connectivity(comms_matrix)


# Choosing Between Distance Model and Signal Model

Our library provides two models for distance-based communication:

## Distance Model
- **Use when**: You need a conceptually simple model with linear degradation
- **Advantages**: Easier to understand, implement, and analyze
- **Disadvantages**: Less physically accurate, no spatial optimization

## Signal Model
- **Use when**: You need physically accurate signal propagation
- **Advantages**: Based on electromagnetic theory, more realistic, computationally efficient
- **Disadvantages**: More complex, may have more parameters to tune

For most research applications requiring realistic wireless communication,
we recommend the Signal Model. The Distance Model is provided primarily
for educational purposes and as a simpler baseline for comparison.