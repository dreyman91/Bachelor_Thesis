
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
import pandas as pd
import time
import numpy as np

# Load Dataframe
df_obs = pd.read_csv("df_obs.csv")

# Extract unique agents
agents = df_obs["Agent"].unique()
steps = df_obs["Step"].unique()

fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.set_title("Agent Movements")

# Dictionary to hold agent markers

agent_markers = {agent: ax.plot([], [], 'o', markersize=8, label=agent)[0] for agent in agents}
collision_markers = []  # red marker for collision
stationary_counters = {}  # tracking how long agents stayed in a place
collision_log = []

# create static  landmark positions
num_landmarks = 3
landmarks = {
    i: ax.add_patch(patches.Rectangle(
        (df_obs.iloc[0]["L1_x"], df_obs.iloc[0]["L1_y"]), 0.2, 0.2, color="lightgray"
    )) for i in range(num_landmarks)  # three dynamic landmarks
}

# Pause/play functionality
is_paused = False


def toggle_pause(event):
    "Toggles animation pause or play"
    global is_paused
    if event.key == " ":
        is_paused = not is_paused


def update(frame):  # Animation frame update
    global is_paused
    if is_paused:
        return

    step = steps[frame]
    step_info = df_obs[df_obs["Step"] == step]

    # Check if step_data is empty
    if step_info.empty:
        print(f"⚠ Warning: No data found for Step {step}")
        return

    print(f"🔹 Step {step}: Updating agent positions")

    agent_positions = {}
    landmark_positions = {}

    # clear previous collision markers
    for marker in collision_markers:
        marker.remove()
    collision_markers.clear()

    for agent in agents:
        agent_data = step_info[(step_info["Agent"] == agent) & (step_info["State"] == "After")]

        print(f"Step {step}, Agent {agent}: Found {len(agent_data)} rows")  # Debugging

        if not agent_data.empty:
            x, y = agent_data.iloc[0][["SP_x", "SP_y"]]
            print(f"✅ Updating {agent} to position ({x}, {y})")
            agent_markers[agent].set_data([x], [y])
            agent_positions[agent] = (float(x), float(y))

            # Track stationary
            if agent in stationary_counters and stationary_counters[agent]["pos"] == (x, y):
                stationary_counters[agent]["pos"] += 0.5  # assume 0.5 sec per frame
            else:
                stationary_counters[agent] = {"pos": (x, y), "time": 0}
            # Stationary time
            if stationary_counters[agent]["time"] > 1:
                ax.text(x, y, f"{stationary_counters[agent]['time']}s", fontsize=8, color="black")

    # Update landmark positions dynamically
        if frame < len(df_obs):
            for i in range(num_landmarks):
                landmark_x, landmark_y = df_obs.iloc[frame]["L1_x"], df_obs.iloc[frame]["L1_y"]
                landmark_x, landmark_y = df_obs.iloc[frame]["L2_x"], df_obs.iloc[frame]["L2_y"]
                landmark_x, landmark_y = df_obs.iloc[frame]["L3_x"], df_obs.iloc[frame]["L3_y"]
                landmarks[i].set_xy((landmark_x, landmark_y))
                landmark_positions[i] = (landmark_x, landmark_y)

        # Detect Collisions

        # check for collisions
        for agent, agent_pos in agent_positions.items():
            for landmark_id, landmark_pos in landmark_positions.items():
                x, y = agent_pos
                marker = ax.plot(x, y, 'yo', markersize=10)[0]
                collision_markers.append(marker)
                ax.text(x, y, f"Collision L{landmark_id}", fontsize=8, color="brown")  # show collision

                # Log collision data
                collision_log.append({
                    "Step": step,
                    "Collision_Location_X": x,
                    "Collision_Location_Y": y,
                    "Agents_Involved": agent,
                    "Collided_with_Landmark": f"L{landmark_id}",
                    "Time": frame * 0.5  # Assuming 0.5 sec per frame
                })

                print(f"⚠ Collision at Step {step}: {agent} collided with Landmark {landmark_id} at ({x}, {y})")

    ax.figure.canvas.draw()
    return list(agent_markers.values()) + collision_markers


# Create animation
animations = [animation.FuncAnimation(fig, update, frames=len(df_obs["Step"].unique()), interval=500, blit=False,
                                     cache_frame_data=False)]

print("✅ Recorded Collisions:", collision_log)

fig.canvas.mpl_connect("key_press_event", toggle_pause)
# Save collision logs as a CSV file
collision_log_df = pd.DataFrame(collision_log)
collision_log_df.to_csv("collision_log.csv", index=False)

plt.legend(loc="upper right", fontsize=8)
plt.show()

print("Unique agents in DF:", df_obs["Agent"].unique())
