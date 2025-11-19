import heapq
# A search
# Block properties: {block_name: (energy_cost, reward)}
block_info = {
    "empty": (1, 0),
    "dirt": (2, 0),
    "stone": (4, 0),
    "deepslate": (10, 0),
    "stone_gold": (4, 5),
    "deepslate_gold": (10, 5),
    "zombie": (50, 0)
}

# Directions: key = move, value = (dx, dy)
directions = {
    'W': (0, -1),  # Up
    'A': (-1, 0),  # Left
    'S': (0, 1),   # Down
    'D': (1, 0)    # Right
}

def agent_logic(game_map, position, energy):
    """
    Finds the shortest path in 7x7 square using cost - reward as edge weights.
    Returns the best initial move ('W', 'A', 'S', 'D') to follow that path.
    """

    width = len(game_map[0])
    height = len(game_map)

    sx, sy = position  # Start position

    # Define 7x7 square bounds
    min_x = max(0, sx - 3)
    max_x = min(width - 1, sx + 3)
    min_y = max(0, sy - 3)
    max_y = min(height - 1, sy + 3)

    # Priority queue: (total_weight, x, y, path)
    frontier = [(0, sx, sy, [])]

    # Visited dictionary to store minimum cost to each cell
    visited = {}

    best_path = None
    best_weight = float('inf')

    while frontier:
        weight, x, y, path = heapq.heappop(frontier)

        # Skip if already visited with lower cost
        if (x, y) in visited and visited[(x, y)] <= weight:
            continue
        visited[(x, y)] = weight

        # Skip the starting cell (no action yet), but store best if it's from a move
        if path and weight < best_weight:
            best_weight = weight
            best_path = path

        # Expand neighbors within the 7x7 square
        for move, (dx, dy) in directions.items():
            nx, ny = x + dx, y + dy

            if not (min_x <= nx <= max_x and min_y <= ny <= max_y):
                continue  # Out of 7x7 region
            if not (0 <= nx < width and 0 <= ny < height):
                continue  # Out of game map bounds

            block = game_map[ny][nx]
            cost, reward = block_info.get(block, (float('inf'), 0))

            edge_weight = cost - reward
            new_path = path + [move]
            heapq.heappush(frontier, (weight + edge_weight, nx, ny, new_path))

    # Return first move in path if found
    if best_path:
        return best_path[0]
    else:
        return None  # No path found