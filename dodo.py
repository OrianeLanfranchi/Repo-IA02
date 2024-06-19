"""
Module for Dodo game initialization and strategies.
"""

from typing import List, Tuple
import time
import random
import Grid

### Initializing
def init_grid(size: int) -> Grid.Grid:
    """
    Initializes the Dodo grid.
    """
    if size <= 0:
        return None
    grid = Grid.size_to_2d_array(size)

    # Initialize: half grid blue, half grid red
    for row in range(0, size + 1):
        for col in range(0, size + 1):
            coor_grid = Grid.axial_to_grid(row, col, size)
            if grid[coor_grid[0]][coor_grid[1]] is not None:
                grid[coor_grid[0]][coor_grid[1]] = Grid.BLUE

    for row in range(-size, 1):
        for col in range(-size, 1):
            coor_grid = Grid.axial_to_grid(row, col, size)
            if grid[coor_grid[0]][coor_grid[1]] is not None:
                grid[coor_grid[0]][coor_grid[1]] = Grid.RED

    for row in range(0, size + 1):
        for col in range(0, size + 1):
            if grid[row][col] is not None:
                coor_ax = Grid.grid_to_axial(row, col, size)
                grid[row][col] = Grid.RED if coor_ax[0] <= -coor_ax[1] else Grid.BLUE

    for row in range(size + 1, size * 2 + 1):
        for col in range(size + 1, size * 2 + 1):
            if grid[row][col] is not None:
                coor_ax = Grid.grid_to_axial(row, col, size)
                grid[row][col] = Grid.BLUE if -coor_ax[0] <= coor_ax[1] else Grid.RED

    # Initialize: middle is 0
    row, col = 0, 0
    while row != -size and col != size:
        coor_grid = Grid.axial_to_grid(row, col, size)
        if grid[coor_grid[0]][coor_grid[1]] is not None:
            grid[coor_grid[0]][coor_grid[1]] = 0

        coor_grid = Grid.axial_to_grid(row - 1, col, size)
        if grid[coor_grid[0]][coor_grid[1]] is not None:
            grid[coor_grid[0]][coor_grid[1]] = 0

        coor_grid = Grid.axial_to_grid(row, col + 1, size)
        if grid[coor_grid[0]][coor_grid[1]] is not None:
            grid[coor_grid[0]][coor_grid[1]] = 0

        row -= 1
        col += 1

    row, col = 0, 0
    while row != size and col != -size:
        coor_grid = Grid.axial_to_grid(row, col, size)
        if grid[coor_grid[0]][coor_grid[1]] is not None:
            grid[coor_grid[0]][coor_grid[1]] = 0

        coor_grid = Grid.axial_to_grid(row + 1, col, size)
        if grid[coor_grid[0]][coor_grid[1]] is not None:
            grid[coor_grid[0]][coor_grid[1]] = 0

        coor_grid = Grid.axial_to_grid(row, col - 1, size)
        if grid[coor_grid[0]][coor_grid[1]] is not None:
            grid[coor_grid[0]][coor_grid[1]] = 0

        row += 1
        col -= 1

    return grid

### Legals
def legals(grid: Grid.Grid, player: Grid.Player) -> List[Grid.ActionDodo]:
    """
    Returns legal actions.
    """
    legals_list = []
    size = (len(grid) - 1) // 2

    for row in range(-size, size + 1):
        for col in range(-size, size + 1):
            coor_grid = Grid.axial_to_grid(row, col, size)

            if grid[coor_grid[0]][coor_grid[1]] == player:
                playable_cells = eval_cells_around(row, col, grid, player, size)

                for cell in playable_cells:
                    legals_list.append(((row, col), cell))

    return legals_list

def eval_cells_around(
    row_ax: int, col_ax: int, grid: Grid.Grid, player: Grid.Player, size: int
) -> List[Grid.Cell]:
    """
    For a given cell, returns available cells around.
    """
    playable_cells = []
    direction = 1 if player == 1 else -1

    if -size <= row_ax + direction <= size:
        # Left diagonal
        coor_grid = Grid.axial_to_grid(row_ax + direction, col_ax, size)
        if grid[coor_grid[0]][coor_grid[1]] == 0:
            playable_cells.append((row_ax + direction, col_ax))

        # In front
        if -size <= col_ax + direction <= size:
            coor_grid = Grid.axial_to_grid(row_ax + direction, col_ax + direction, size)
            if grid[coor_grid[0]][coor_grid[1]] == 0:
                playable_cells.append((row_ax + direction, col_ax + direction))

    # Right diagonal
    if -size <= col_ax + direction <= size:
        coor_grid = Grid.axial_to_grid(row_ax, col_ax + direction, size)
        if grid[coor_grid[0]][coor_grid[1]] == 0:
            playable_cells.append((row_ax, col_ax + direction))

    return playable_cells

### Final
def is_final(grid: Grid.Grid) -> bool:
    """
    Check if the game is in a final state.
    """
    return len(legals(grid, 1)) == 0 or len(legals(grid, 2)) == 0

def is_final_player(grid: Grid.Grid, player: Grid.Player) -> bool:
    """
    Check if the game is in a final state for a specific player.
    """
    return len(legals(grid, player)) == 0

### Score
def score(state: Grid.State) -> float:
    """
    Returns the score of the game state.
    """
    grid = Grid.state_to_grid(state)
    if is_final_player(grid, Grid.RED):
        return 1
    if is_final_player(grid, Grid.BLUE):
        return -1
    return 0

# Strategy
def strategy_user(state: Grid.State, player: Grid.Player) -> Grid.ActionDodo:
    """
    User strategy for selecting an action.
    """
    selection = False
    actions = legals(Grid.state_to_grid(state), player)

    print("Player:", player, "\nAvailable actions:")
    for action in actions:
        print(action)

    while not selection:
        row_start = int(input("Choose a row (start): "))
        col_start = int(input("Choose a col (start): "))
        row_end = int(input("Choose a row (end): "))
        col_end = int(input("Choose a col (end): "))
        if ((row_start, col_start), (row_end, col_end)) in actions:
            player_action = ((row_start, col_start), (row_end, col_end))
            selection = True

    return player_action

def strategy_random(
    env: Grid.Environment, state: Grid.State, player: Grid.Player, _: Grid.Time
) -> Tuple[Grid.Environment, Grid.Action]:
    """
    Random strategy for selecting an action.
    """
    return env, random.choice(legals(Grid.state_to_grid(state), player))

def evaluate(grid: Grid.Grid, player: Grid.Player) -> float:
    """
    Evaluate the game state for a specific player.
    """
    opponent = player % 2 + 1
    player_legal_moves = len(legals(grid, player))
    opponent_legal_moves = len(legals(grid, opponent))
    return opponent_legal_moves - player_legal_moves

# Negamax
def nega_max(state: Grid.State, player: Grid.Player, depth: int) -> float:
    """
    Negamax algorithm for evaluating game states.
    """
    grid = Grid.state_to_grid(state)
    if depth == 0 or is_final_player(grid, player):
        return evaluate(grid, player)

    score_tmp = float('-inf')
    moves = legals(grid, player)

    for move in moves:
        new_state = play(state, player, move)
        cur = -nega_max(new_state, player % 2 + 1, depth - 1)
        score_tmp = max(score_tmp, cur)

    return score_tmp

def nega_max_action(
    state: Grid.State, player: Grid.Player, depth: int
) -> Tuple[float, Grid.ActionDodo]:
    """
    Get the best action using the Negamax algorithm.
    """
    best_score = float('-inf')
    best_action = None

    list_actions = legals(Grid.state_to_grid(state), player)
    for possible_action in list_actions:
        new_state = play(state, player, possible_action)
        eval_score = -nega_max(new_state, player % 2 + 1, depth - 1)
        if eval_score > best_score:
            best_score = eval_score
            best_action = possible_action

    return best_score, best_action

def strategy_nega_max(
    env: Grid.Environment, state: Grid.State, player: Grid.Player, _: Grid.Time, depth: int = 3
) -> Tuple[Grid.Environment, Grid.ActionDodo]:
    """
    Strategy using the Negamax algorithm.
    """
    _, best_action = nega_max_action(state, player, depth)
    return "dodo", best_action

# Negamax pruning
def nega_max_alpha_beta(
    state: Grid.State, player: Grid.Player, depth: int, alpha: float, beta: float
) -> float:
    """
    Negamax algorithm with alpha-beta pruning for evaluating game states.
    """
    grid = Grid.state_to_grid(state)
    if depth == 0 or is_final_player(grid, player):
        return evaluate(grid, player)

    score_tmp = float('-inf')
    moves = legals(grid, player)

    for move in moves:
        new_state = play(state, player, move)
        cur = -nega_max_alpha_beta(new_state, player % 2 + 1, depth - 1, -beta, -alpha)
        score_tmp = max(score_tmp, cur)
        alpha = max(alpha, cur)
        if alpha >= beta:
            break

    return score_tmp

def nega_max_alpha_beta_action(
    state: Grid.State, player: Grid.Player, depth: int
) -> Tuple[float, Grid.ActionDodo]:
    """
    Get the best action using the Negamax algorithm with alpha-beta pruning.
    """
    best_score = float('-inf')
    best_action = None
    alpha = float('-inf')
    beta = float('inf')

    list_actions = legals(Grid.state_to_grid(state), player)
    for possible_action in list_actions:
        new_state = play(state, player, possible_action)
        eval_score = -nega_max_alpha_beta(
            new_state, player % 2 + 1, depth - 1, alpha, beta
        )
        if eval_score > best_score:
            best_score = eval_score
            best_action = possible_action

    return best_score, best_action

def strategy_nega_max_alpha_beta(
    env: Grid.Environment, state: Grid.State, player: Grid.Player, time_left: Grid.Time, depth: int = 1
) -> Tuple[Grid.Environment, Grid.ActionDodo]:
    """
    Strategy using the Negamax algorithm with alpha-beta pruning.
    """
    _, best_action = nega_max_alpha_beta_action(state, player, depth)
    print(time_left)
    return env, best_action

### Evaluation functions
def number_of_legals(state: Grid.State, player: Grid.Player) -> float:
    """
    Returns number of legal moves left for a given player.
    """
    grid = Grid.state_to_grid(state)
    if player == Grid.RED:
        return len(legals(grid, Grid.RED))
    return len(legals(grid, Grid.BLUE))

### Play
def play(state: Grid.State, player: Grid.Player, action: Grid.ActionDodo) -> Grid.State:
    """
    Updates the grid with a player's move.
    """
    grid = Grid.state_to_grid(state)
    coor_grid_start = Grid.axial_to_grid(action[0][0], action[0][1], (len(grid) - 1) // 2)
    coor_grid_end = Grid.axial_to_grid(action[1][0], action[1][1], (len(grid) - 1) // 2)
    grid[coor_grid_start[0]][coor_grid_start[1]] = 0
    grid[coor_grid_end[0]][coor_grid_end[1]] = player
    return Grid.grid_to_state(grid)

### Game
def game_play(size: int, strategy_red: Grid.Strategy, strategy_blue: Grid.Strategy):
    """
    Simulate a game play between two strategies.
    """
    real_size = size - 1  # Adapting to convention

    # Game structure
    state: Grid.State = Grid.grid_to_state(init_grid(real_size))
    env_red = []
    env_blue = []

    # Best opening moves on a size 3 grid
    squares_to_secure_red = []
    squares_to_secure_blue = []

    squares_to_secure_red.append(((-1, -1), (0, 0)))
    squares_to_secure_red.append(
        ((-(real_size // 2) - 1, (real_size // 2) - 1), (-(real_size // 2), real_size // 2))
    )
    squares_to_secure_red.append(
        (((real_size // 2) - 1, (-(real_size // 2) - 1)), (real_size // 2, -(real_size // 2)))
    )

    squares_to_secure_blue.append(((1, 1), (0, 0)))
    squares_to_secure_blue.append(
        ((-(real_size // 2) + 1, (real_size // 2) + 1), (-(real_size // 2), real_size // 2))
    )
    squares_to_secure_blue.append(
        (((real_size // 2) + 1, (-real_size // 2) + 1), (real_size // 2, -real_size // 2))
    )

    env_red.append(squares_to_secure_red)
    env_blue.append(squares_to_secure_blue)

    current_player = Grid.RED
    while not is_final_player(Grid.state_to_grid(state), current_player):
        if current_player == Grid.RED:
            env_red, action = strategy_red(env_red, state, current_player, time)
            state = play(state, current_player, action)
            current_player = Grid.BLUE

        elif current_player == Grid.BLUE:
            env_blue, action = strategy_blue(env_blue, state, current_player, time)
            state = play(state, current_player, action)
            current_player = Grid.RED

    return score(state)

### Main function
def main():
    """
    Main function to run the game simulations.
    """
    start = time.time()
    list_scores = []
    for i in range(100):
        print("Partie:", i)
        list_scores.append(game_play(4, strategy_nega_max_alpha_beta, strategy_random))

    print(time.time() - start)
    print(list_scores.count(1), "/", len(list_scores))

if __name__ == "__main__":
    main()
