
import numpy as np
import random
#import matplotlib.pyplot as plt
two_gold_targets = None
gold_phase = None
gold_path = None


ACTIONS = {
    'W': (0, -1),
    'A': (-1, 0),
    'S': (0, 1),
    'D': (1, 0)
}

GAMMA = 0.9
MOVE_PENALTY = 0
VISIT_PENALTY_WEIGHT = 0.01

BLOCK_REWARD = {
    'empty': 10,
    'dirt': 0,
    'stone': 0,
    'deepslate': 0,
    'stone_gold': 90,
    'deepslate_gold': 50,
    'zombie': -80
}

BLOCK_COST = {
    'empty': 1,
    'dirt': 2,
    'stone': 4,
    'deepslate': 10,
    'gold': 4,
    'deepslate_gold': 10,
    'zombie': 80
}

visit_count_map = None
blacklist = None

def mark_zombie_danger(game_map):
    width, height = len(game_map), len(game_map[0])
    danger = [[False]*height for _ in range(width)]
    for x in range(width):
        for y in range(height):
            if game_map[x][y] == "zombie":
                danger[x][y] = True
                for dx, dy in ACTIONS.values():
                    nx, ny = x+dx, y+dy
                    if 0 <= nx < width and 0 <= ny < height:
                        danger[nx][ny] = True
    return danger

def find_connected_region(game_map, start, visited):
    stack = [start]
    region = []
    width, height = len(game_map), len(game_map[0])
    while stack:
        x, y = stack.pop()
        if visited[x][y]:
            continue
        visited[x][y] = True
        region.append((x, y))
        for dx, dy in ACTIONS.values():
            nx, ny = x+dx, y+dy
            if 0 <= nx < width and 0 <= ny < height:
                if game_map[nx][ny] == "empty" and not visited[nx][ny]:
                    stack.append((nx, ny))
    return region

def build_blacklist(game_map, agent_pos):
    width, height = len(game_map), len(game_map[0])
    visited = [[False]*height for _ in range(width)]

    def bfs(start, is_empty):
        queue = [start]
        region = []
        sx, sy = start
        visited[sx][sy] = True
        while queue:
            x, y = queue.pop()
            region.append((x, y))
            for dx, dy in ACTIONS.values():
                nx, ny = x+dx, y+dy
                if 0 <= nx < width and 0 <= ny < height and not visited[nx][ny]:
                    cond = (game_map[nx][ny] == "empty") if is_empty else (game_map[nx][ny] != "empty" and game_map[nx][ny] != "zombie")
                    if cond:
                        visited[nx][ny] = True
                        queue.append((nx, ny))
        return region

    # 1. Label regions: empty and non-empty (zombies excluded)
    empty_regions = []
    nonempty_regions = []
    region_id_map = [[None]*height for _ in range(width)]
    region_nodes = []

    for x in range(width):
        for y in range(height):
            if not visited[x][y]:
                if game_map[x][y] == "empty":
                    region = bfs((x, y), is_empty=True)
                    if region:
                        idx = len(region_nodes)
                        for px, py in region:
                            region_id_map[px][py] = idx
                        empty_regions.append(region)
                        region_nodes.append(('empty', region))
                elif game_map[x][y] != "zombie":
                    region = bfs((x, y), is_empty=False)
                    if region:
                        idx = len(region_nodes)
                        for px, py in region:
                            region_id_map[px][py] = idx
                        if len(region) > 200:
                            nonempty_regions.append(region)
                            region_nodes.append(('nonempty', region))
                        else:
                            region_nodes.append(('small_nonempty', region))

    # 2. Build metagraph of adjacency between regions
    from collections import defaultdict, deque

    graph = defaultdict(set)

    for idx, (_, region) in enumerate(region_nodes):
        for x, y in region:
            for dx, dy in ACTIONS.values():
                nx, ny = x + dx, y + dy
                if 0 <= nx < width and 0 <= ny < height:
                    nid = region_id_map[nx][ny]
                    if nid is not None and nid != idx:
                        graph[idx].add(nid)

    # 3. Determine which empty regions are CRITICAL bridges
    def is_critical(empty_idx):
        temp_graph = {k: set(v) for k, v in graph.items()}
        # remove empty region and all its edges
        for neighbor in temp_graph.get(empty_idx, []):
            temp_graph[neighbor].discard(empty_idx)
        temp_graph.pop(empty_idx, None)

        # get all remaining large non-empty region indices
        targets = [i for i, (typ, _) in enumerate(region_nodes) if typ == 'nonempty']
        if len(targets) <= 1:
            return False  # trivially connected

        # check if they are all still reachable from first one
        start = targets[0]
        visited = set()
        queue = deque([start])
        while queue:
            node = queue.popleft()
            if node in visited:
                continue
            visited.add(node)
            for nei in temp_graph.get(node, []):
                queue.append(nei)

        return not all(t in visited for t in targets)

    # 4. Collect all empty regions adjacent to zombie
    danger_zone = set()
    for x in range(width):
        for y in range(height):
            if game_map[x][y] == "zombie":
                for dx, dy in ACTIONS.values():
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < width and 0 <= ny < height and game_map[nx][ny] == "empty":
                        danger_zone.add(region_id_map[nx][ny])

    forbidden = set()
    for i in danger_zone:
        if i is not None and region_nodes[i][0] == 'empty':
            if not is_critical(i):
                forbidden.update(region_nodes[i][1])

    # 5. Subtract agent’s safe region (unconditionally allowed)
    visited2 = [[False]*height for _ in range(width)]
    safe_region = find_connected_region(game_map, agent_pos, visited2)
    forbidden -= set(safe_region)

    # 6. Add margin around forbidden blocks
    blacklist_set = set(forbidden)
    for x, y in forbidden:
        for dx, dy in ACTIONS.values():
            nx, ny = x+dx, y+dy
            if 0 <= nx < width and 0 <= ny < height:
                blacklist_set.add((nx, ny))

    return blacklist_set

def visualize_forbidden_area(game_map, blacklist):
    width, height = len(game_map), len(game_map[0])
    m = np.zeros((width, height))
    for x, y in blacklist:
        m[x][y] = 1
    plt.figure(figsize=(6, 5))
    plt.imshow(m.T, cmap='Reds', origin='upper')
    plt.title("Forbidden Area")
    plt.colorbar()
    plt.show()

def visualize_value_map(V, title="Value Map"):
    plt.figure(figsize=(6, 5))
    plt.imshow(V.T, cmap='plasma', origin='upper')
    plt.title(title)
    plt.colorbar()
    plt.show()

# ------------RECORD---------------
# constant: blacklist = 200
# max iter = 23, 

# constant: max iter = 24
# blacklist = 180
# blacklist = 190
# blacklist = 200
# blacklist = 210
# blacklist = 220
# all score = 568

# constant: max iter = 25
# blacklist = 180
# blacklist = 190
# blacklist = 200
# blacklist = 210
# blacklist = 220
# blacklist = 230
# all score = 568


def value_iteration(game_map, visit_count, blacklist, gamma=GAMMA, threshold=1e-4, max_iter=25):
    width, height = len(game_map), len(game_map[0])
    danger = mark_zombie_danger(game_map)
    V = np.zeros((width, height))
    policy = [['' for _ in range(height)] for _ in range(width)]

    for _ in range(max_iter):
        delta = 0
        for x in range(width):
            for y in range(height):
                if (x, y) in blacklist:
                    continue
                best_val = float('-inf')
                best_act = None
                for action, (dx, dy) in ACTIONS.items():
                    nx, ny = x+dx, y+dy
                    if 0 <= nx < width and 0 <= ny < height:
                        if danger[nx][ny] or (nx, ny) in blacklist:
                            continue
                        block = game_map[nx][ny]
                        reward = BLOCK_REWARD.get(block, 0)
                        cost = BLOCK_COST.get(block, 1)
                        penalty = VISIT_PENALTY_WEIGHT * visit_count[nx][ny]
                        val = reward - cost - penalty + gamma * V[nx][ny]
                        if val > best_val:
                            best_val = val
                            best_act = action
                if best_act is not None:
                    delta = max(delta, abs(V[x][y] - best_val))
                    V[x][y] = best_val
                    policy[x][y] = best_act
        if delta < threshold:
            break
    return policy, V

def get_escape_action(game_map, position, blacklist):
    """
    When agent is stuck, choose the action that maximizes the
    Manhattan distance from zombies within radius <= 3.
    """
    x, y = position
    width, height = len(game_map), len(game_map[0])

    # Collect zombie positions within dist <= 3
    zombies = []
    for i in range(width):
        for j in range(height):
            if game_map[i][j] == "zombie":
                if abs(i - x) + abs(j - y) <= 3:
                    zombies.append((i, j))

    # If no zombies nearby, fallback to random
    if not zombies:
        return random.choice(["W", "A", "S", "D"])

    best_act = None
    best_score = -float('inf')

    for act, (dx, dy) in ACTIONS.items():
        nx, ny = x + dx, y + dy

        # must stay inside map
        if not (0 <= nx < width and 0 <= ny < height):
            continue
        # cannot walk into forbidden region
        if (nx, ny) in blacklist:
            continue

        # compute minimum distance to any nearby zombie
        score = min(abs(nx - zx) + abs(ny - zy) for zx, zy in zombies)

        # maximize this score
        if score > best_score:
            best_score = score
            best_act = act

    # fallback if no valid escape found
    return best_act

def agent_logic(game_map, position, energy):
    global visit_count_map, blacklist
    global two_gold_targets, gold_phase, gold_path

    width, height = len(game_map), len(game_map[0])
    if visit_count_map is None or visit_count_map.shape != (width, height):
        visit_count_map = np.zeros((width, height), dtype=int)
    if blacklist is None:
        blacklist = build_blacklist(game_map, position)

    x, y = position
    if (x, y) in blacklist:
        return "I"

    # -------- Handle two gold phases --------
    if two_gold_targets and gold_phase:
        (g1x, g1y), (g2x, g2y) = two_gold_targets

        if gold_phase == "back":
            # Go back to previous position (gold_path)
            for dir, (dx, dy) in ACTIONS.items():
                if (x + dx, y + dy) == gold_path:
                    gold_phase = "dig2"
                    return dir

        elif gold_phase == "dig2":
            # Move to second gold (which is still gold)
            target = (g2x, g2y) if game_map[g2x][g2y] == "stone_gold" else (g1x, g1y)
            for dir, (dx, dy) in ACTIONS.items():
                if (x + dx, y + dy) == target:
                    gold_phase = "return"
                    return dir

        elif gold_phase == "return":
            # Return to first gold (now empty)
            for dir, (dx, dy) in ACTIONS.items():
                if (x + dx, y + dy) == gold_path:
                    two_gold_targets = None
                    gold_phase = None
                    gold_path = None
                    return dir

    # -------- Detect 3x3 clean zone & adjacent stone_gold --------
    adjacent_gold = []
    valid = True
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            nx, ny = x + dx, y + dy
            if not (0 <= nx < width and 0 <= ny < height):
                continue
            block = game_map[nx][ny]
            if (nx, ny) in blacklist:
                valid = False
            if block not in ('stone', 'empty', 'stone_gold'):
                valid = False

    for direction, (dx, dy) in ACTIONS.items():
        nx, ny = x + dx, y + dy
        if 0 <= nx < width and 0 <= ny < height and game_map[nx][ny] == "stone_gold":
            adjacent_gold.append(((nx, ny), direction))

    # -------- Single stone_gold --------
    if valid and len(adjacent_gold) == 1:
        (_, direction) = adjacent_gold[0]
        #print(f"Single stone_gold detected, moving to direction {direction}")
        return direction

    # -------- Two stone_gold detected: plan multi-step route --------
    if valid and len(adjacent_gold) == 2:
        ((g1x, g1y), d1), ((g2x, g2y), d2) = adjacent_gold
        two_gold_targets = [(g1x, g1y), (g2x, g2y)]

        # Let value iteration decide which gold is better
        policy, V = value_iteration(game_map, visit_count_map, blacklist)
        if V[g1x][g1y] >= V[g2x][g2y]:
            chosen = (g1x, g1y)
            move_dir = d1
        else:
            chosen = (g2x, g2y)
            move_dir = d2

        gold_phase = "back"
        gold_path = (x, y)
        #print("Two gold situation detected")
        return move_dir

    # -------- Default logic --------
    policy, V = value_iteration(game_map, visit_count_map, blacklist)
    visit_count_map[x][y] += 10

    if visit_count_map[x][y] >= 40:
        action = get_escape_action(game_map, position, blacklist)
        print(f"Taking escape action {action}")
        return action if action else random.choice(["W", "A", "S", "D"])

    move = policy[x][y]
    return move if move else "I"

