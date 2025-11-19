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
MOVE_PENALTY = 0.5                   # Small penalty every time the agent moves
VISIT_PENALTY_WEIGHT = 0.2           # Penalty increases for frequently visited cells

# --- Reward and Cost Configurations for Each Block Type ---
BLOCK_REWARD = {
    'empty': 0,                     # No reward
    'dirt': 0,
    'stone': 0,
    'deepslate': 0,
    'stone_gold': 100,              # Very valuable target
    'deepslate_gold': 60,           # Also valuable
    'zombie': 0                     # No reward (but penalized elsewhere)
}

BLOCK_COST = {
    'empty': 15,                    # Cost to traverse this block
    'dirt': 2,
    'stone': 4,
    'deepslate': 10,
    'gold': 4,
    'deepslate_gold': 10,
    'zombie': 40                    # Not used directly due to danger avoidance
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


# ---------------------------------------------------------------
# Plot net reward for each cell using matplotlib (for debugging/visualization)
# ---------------------------------------------------------------
def visualize_net_reward_map(game_map, visit_count, danger):
    width = len(game_map)
    height = len(game_map[0])
    net_reward_map = np.zeros((width, height))     # Matrix of net rewards

    for x in range(width):
        for y in range(height):
            block = game_map[x][y]                          # Get block type
            cost = BLOCK_COST.get(block, 1)                 # Movement/mining cost
            reward = BLOCK_REWARD.get(block, 0)             # Reward value
            visit_penalty = VISIT_PENALTY_WEIGHT * visit_count[x][y]  # Penalty for revisits
            net_reward = reward - cost - MOVE_PENALTY - visit_penalty  # Combined net reward
            if danger[x][y]:
                net_reward -= 100                           # Strong penalty for danger
            net_reward_map[x][y] = net_reward               # Save result

    # Plot the matrix as a heatmap
    plt.figure(figsize=(8, 6))
    plt.imshow(net_reward_map.T, cmap='viridis', origin='lower')  # Transposed for orientation
    plt.colorbar(label='Net Reward')
    plt.title('Initial Net Reward Map (Iteration 1)')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.xticks(np.arange(width))
    plt.yticks(np.arange(height))
    plt.grid(False)
    plt.show()


# ---------------------------------------------------------------
# Value Iteration (core MDP solver)
# Computes best move for each cell by iteratively propagating rewards
# ---------------------------------------------------------------
def value_iteration(game_map, energy, visit_count, gamma=GAMMA, threshold=1e-3):
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
    #visualize_net_reward_map(game_map, visit_count, danger)  # Visualize once
    return policy                                             # Return best moves per cell


# ---------------------------------------------------------------
# Main entrypoint for each agent turn (called once per tick)
# ---------------------------------------------------------------
def agent_logic(game_map, position, energy):
    global visit_count_map

    width = len(game_map)
    height = len(game_map[0])

    # Initialize global visit map if not already initialized or map size changed
    if visit_count_map is None or visit_count_map.shape != (width, height):
        visit_count_map = np.zeros((width, height), dtype=int)

    x, y = position                                           # Agent's current position

    policy = value_iteration(game_map, energy, visit_count_map)  # Run MDP planning

    visit_count_map[x][y] += 20                               # Mark current cell as visited

    move = policy[x][y]                                       # Get best move for current cell
    return move if move else 'I'                              # If no move found, idle
