# VERSION 2
import numpy as np                   # For efficient array/matrix operations
import random                        # For future use (e.g., random tie-breaking)
import matplotlib.pyplot as plt      # For plotting the net reward heatmap

# --- Movement Actions and Config ---
ACTIONS = {
    'W': (0, -1),                    # Move Up (North)
    'A': (-1, 0),                    # Move Left (West)
    'S': (0, 1),                     # Move Down (South)
    'D': (1, 0)                      # Move Right (East)
}

GAMMA = 0.9                          # Discount factor for future rewards
MOVE_PENALTY = 0                  # Small penalty every time the agent moves
VISIT_PENALTY_WEIGHT = 0.01           # Penalty increases for frequently visited cells

# --- Reward and Cost Configurations for Each Block Type ---
BLOCK_REWARD = {
    'empty': 0,                     # No reward
    'dirt': 0,
    'stone': 0,
    'deepslate': 0,
    'stone_gold': 55,              # Very valuable target
    'deepslate_gold': 50,           # Also valuable
    'zombie': -80                     # No reward (but penalized elsewhere)
}

BLOCK_COST = {
    'empty': 2,                    # Cost to traverse this block
    'dirt': 1,
    'stone': 4,
    'deepslate': 10,
    'gold': 4,
    'deepslate_gold': 10,
    'zombie': 80                    # Not used directly due to danger avoidance
}

visit_count_map = None              # Global tracker of visit frequency for all cells


# ---------------------------------------------------------------
# Construct a "danger map" where zombies and their neighbors are marked unsafe
# ---------------------------------------------------------------
def mark_zombie_danger(game_map):
    width = len(game_map)                           # Width of the map
    height = len(game_map[0])                       # Height of the map
    danger = [[False for _ in range(height)] for _ in range(width)]  # Initialize map

    for x in range(width):
        for y in range(height):
            if game_map[x][y] == 'zombie':          # If this tile has a zombie
                danger[x][y] = True                 # Mark the tile itself as dangerous
                for dx, dy in ACTIONS.values():     # Also mark adjacent tiles
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < width and 0 <= ny < height:
                        danger[nx][ny] = True
    return danger                                   # Return the full danger mask


def visualize_value_map(V, title="Value Map"):
    """Visualize the actual value map V with Y-axis reversed (row 0 on top)."""
    plt.figure(figsize=(8, 6))
    plt.imshow(V.T, cmap='plasma', origin='lower')       # Transpose to match game layout
    plt.colorbar(label='Value')                          # Show color scale
    plt.title(title)                                     # Add title
    plt.xlabel('X')                                      # X-axis label
    plt.ylabel('Y')                                      # Y-axis label
    plt.xticks(np.arange(V.shape[0]))                    # Set X ticks
    plt.yticks(np.arange(V.shape[1]))                    # Set Y ticks
    plt.gca().invert_yaxis()                             # 🔁 Flip Y axis
    plt.grid(False)                                      # Optional: remove grid
    plt.show()




# ---------------------------------------------------------------
# Value Iteration (core MDP solver)
# Computes best move for each cell by iteratively propagating rewards
# ---------------------------------------------------------------
def value_iteration(x1, y1, game_map, visit_count, gamma=GAMMA, threshold=1e-3):
    width = len(game_map)
    height = len(game_map[0])
    danger = mark_zombie_danger(game_map)                     # Mark danger zones

    V = np.zeros((width, height))                             # Initialize value map
    policy = [['' for _ in range(height)] for _ in range(width)]  # Best action for each cell

    iteration = 0                                             # Iteration counter

    while True:
        delta = 0                                             # Tracks value change per iteration
        iteration += 1


        for x in range(width):
            for y in range(height):

                if abs(x - x1) + abs(y - y1) == 1:
                    V[x][y] += 200

                best_value = float('-inf')                    # Track best value found
                best_action = None

                for action, (dx, dy) in ACTIONS.items():      # Try all directions
                    nx, ny = x + dx, y + dy                   # Compute neighbor coordinates

                    if 0 <= nx < width and 0 <= ny < height:  # Check if inside bounds
                        if danger[nx][ny]:                    # Skip if dangerous
                            continue

                        nb_block = game_map[nx][ny]           # Block type at neighbor
                        nb_cost = BLOCK_COST.get(nb_block, 1) # Cost of moving there
                        nb_reward = BLOCK_REWARD.get(nb_block, 0)  # Reward in neighbor
                        nb_visit_penalty = VISIT_PENALTY_WEIGHT * visit_count[nx][ny]  # Penalty

                        nb_net_reward = nb_reward - nb_cost - MOVE_PENALTY - nb_visit_penalty

                        if danger[nx][ny]:
                            nb_net_reward -= 100              # Reinforce penalty if needed

                        value = nb_net_reward + gamma * V[nx][ny]  # Bellman equation

                        if value > best_value:                # If this is the best so far
                            best_value = value
                            best_action = action              # Track direction

                if best_action is not None:
                    delta = max(delta, abs(V[x][y] - best_value))  # Track max change
                    V[x][y] = best_value                          # Update value
                    policy[x][y] = best_action                    # Save best move

        if delta < threshold:                                # Stop when values converge
            break
    #visualize_value_map(V, title=f"Value Map After Iteration {iteration}")
    # Visualize once
    return policy                                             # Return best moves per cell


def agent_logic(game_map, position, energy):
    global visit_count_map

    width = len(game_map)
    height = len(game_map[0])

    # Initialize global visit map if not already initialized or map size changed
    if visit_count_map is None or visit_count_map.shape != (width, height):
        visit_count_map = np.zeros((width, height), dtype=int)

    x1, y1 = position                                           # Agent's current position

    policy = value_iteration(x1, y1, game_map, visit_count_map)  # Run MDP planning

    visit_count_map[x1][y1] += 10                             # Mark current cell as visited
    if visit_count_map[x1][y1] >= 40:
        options = ['W', 'A', 'S', 'D']      # A list of movement directions
        option = random.choice(options) 
        return option

    move = policy[x1][y1]                                       # Get best move for current cell
    return move if move else 'I'                              # If no move found, idle
   