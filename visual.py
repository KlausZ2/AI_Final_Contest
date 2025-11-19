import json
import numpy as np
import matplotlib.pyplot as plt
import agent_logic

# Load game map from JSON
with open("converted_game_map.json") as f:
    game_map = json.load(f)

# Create zero visit count map
visit_count = np.zeros((len(game_map), len(game_map[0])), dtype=int)

# Run value iteration once to convergence
V = np.zeros((len(game_map), len(game_map[0])))
policy = agent_logic.value_iteration(game_map, energy=100, visit_count=visit_count)

# Show converged value map
fig, ax = plt.subplots(figsize=(10, 6))
cax = ax.imshow(V.T, cmap='RdYlGn', origin='lower')
fig.colorbar(cax)
ax.set_title("Converged Value Map")
plt.show()
