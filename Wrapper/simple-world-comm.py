"""
=============================
Interactive iplot Animation for Wrapped Observations
=============================
Author: Your Name
Description:
This script visualizes the movement and interaction of agents
in the Simple Spread environment with a limited observation wrapper.

- Uses Plotly's `iplot` for smooth animations.
- Tracks agent positions over multiple steps.
- Highlights how limiting observations affects agent decisions.
- Helps in understanding agent behavior and interactions.
"""

# Import required libraries
from Wrapper.observabilty_wrapper import LimitedObservabilityWrapper
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pettingzoo.mpe import simple_spread_v3
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter

# Initialize wrapped environment
env = simple_spread_v3.parallel_env()
wrapped_env = LimitedObservabilityWrapper(env, hide_agents=True, hide_landmarks=False)

# Reset environment to get initial observations
observations = wrapped_env.reset()

# Define tracking storage
agent_positions = {agent: [] for agent in wrapped_env.agents}  # Store positions over time
landmark_positions = []  # Store landmark positions

# Number of steps for animation
num_steps = 25

# Run environment steps and log positions
for step in range(num_steps):
    # Get current agent positions
    for agent in wrapped_env.agents:
        agent_positions[agent].append(observations[agent][:2])  # Only (x, y) position

    # Store landmark positions (only once, since landmarks are static)
        landmark_positions.append([observations[agent][4:6] for agent in wrapped_env.agents])

    # Agents take random actions
    actions = {agent: wrapped_env.action_space(agent).sample() for agent in wrapped_env.agents}
    observations, rewards, terminations, truncations, infos = wrapped_env.step(actions)

landmark_positions = np.array(landmark_positions)

# Convert lists to NumPy arrays for easy plotting
for agent in agent_positions:
    agent_positions[agent] = np.array(agent_positions[agent])


# ------------------
# Matplotlib Animation
# -------------------

fig, ax = plt.subplots(figsize=(12, 8))


# Plot dynamic agent movements
agent_plots = {}
for agent in agent_positions:
    agent_plots[agent], = ax.plot([], [], 'o-', label=agent) # o- for marker line

# Plot static landmarks
landmark_plot, = ax.plot([],[], 'rD', markersize=8, label='Landmarks') # Red Diamond markers

# Define update function for animation
def update(frame):

    if frame >= len(landmark_positions):
        frame = len(landmark_positions) - 1 # prevent index error

    for agent, plot in agent_plots.items():
        positions = agent_positions[agent][:frame+1] # Get positions up to current frame
        plot.set_data(positions[:, 0],  positions[:, 1]) # Update agent position


    # Update landmark positions dynamically
    current_landmarks = landmark_positions[frame] # Get landmark for current step
    if current_landmarks.ndim == 1:
        current_landmarks = [current_landmarks]
    landmark_x = [pos[0] for pos in current_landmarks]
    landmark_y = [pos[1] for pos in current_landmarks]
    landmark_plot.set_data(landmark_x, landmark_y) # Update landmark positions

    return list(agent_plots.values()) + [landmark_plot]

# Set plot limits
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.set_title("Dynamic Agent Movement with Limited Observations")
ax.set_xlabel("X Position")
ax.set_ylabel("Y Position")
ax.grid(True)
ax.legend()

# Create animation
ani = animation.FuncAnimation(fig, update, frames=num_steps, interval=200, blit=False)
anim_ref = ani



ani.save("agent_movement.mp4", writer=FFMpegWriter(fps=10))

plt.show()

