"""Module for Gopher game AI strategies and utilities."""

import time
import random
from typing import Callable, List, Tuple, Union
import Grid
import openai
from openai import OpenAI


def memoize(
    f: Callable[
        [Grid.State, Grid.Player, int, float, float],
        Tuple[Grid.Score, Grid.ActionGopher],
    ]
) -> Callable[[Grid.State, Grid.Player, int], Tuple[Grid.Score, Grid.ActionGopher]]:
    """Memoize function results to improve performance."""
    cache = {}  # closure

    def g(
        state: Grid.State, player: Grid.Player, depth: int, alpha: float, beta: float
    ):
        state_tuple = Grid.state_to_tuple(state)
        if state_tuple in cache:
            return cache[state_tuple]

        val = f(state, player, depth, alpha, beta)
        cache[state_tuple] = val

        return val

    return g


def memoize_old(
    f: Callable[[Grid.State, Grid.Player, int], Tuple[Grid.Score, Grid.ActionGopher]]
) -> Callable[[Grid.State, Grid.Player, int], Tuple[Grid.Score, Grid.ActionGopher]]:
    """Memoize function results to improve performance."""
    cache = {}  # closure

    def g(state: Grid.State, player: Grid.Player, depth: int):
        state_tuple = Grid.state_to_tuple(state)
        if state_tuple in cache:
            return cache[state_tuple]

        val = f(state, player, depth)
        cache[state_tuple] = val

        return val

    return g


def memoize_ab(
    f: Callable[[Grid.State, Grid.Player, int], Tuple[Grid.Score, Grid.ActionGopher]]
) -> Callable[[Grid.State, Grid.Player, int], Tuple[Grid.Score, Grid.ActionGopher]]:
    """Memoize function results to improve performance."""
    cache = {}  # closure

    def g(state: Grid.State, player: Grid.Player, depth: int):
        state_tuple = Grid.state_to_tuple(state)
        if state_tuple in cache:
            return cache[state_tuple]

        val = f(state, player, depth)
        cache[state_tuple] = val

        return val

    return g


def init_grid(size: int) -> Grid.Grid:
    """Initialize the grid."""
    return Grid.size_to_2d_array(size)


def legals(grid: Grid.Grid, player: Grid.Player) -> List[Grid.ActionGopher]:
    """Return a list of legal actions for the player."""
    size = (len(grid) - 1) // 2
    list_legals = []
    opponent = player % 2 + 1

    if Grid.is_grid_empty(grid):
        for row_idx, row in enumerate(grid):
            for col_idx, cell in enumerate(row):
                if cell is not None:
                    list_legals.append(Grid.grid_to_axial(row_idx, col_idx, size))
        return list_legals


    for row in range(-size, size + 1):
        for col in range(-size, size + 1):
            coor_grid = Grid.axial_to_grid(row, col, size)
            if (grid[coor_grid[0]][coor_grid[1]] == 0) and (
                eval_cells_around(row, col, grid, player, opponent)
            ):
                list_legals.append((row, col))

    return list_legals


def eval_cells_around(
    row_ax: int, col_ax: int, grid: Grid.Grid, player: Grid.Player, opponent: Grid.Player
) -> bool:
    """Evaluate cells around the given coordinates to determine if the move is valid."""
    enemy = 0

    for i in range(-1, 2):
        for j in range(-1, 2):
            a = (i == 0 and j == 0)
            b = (i == -1 and j == 1)
            c = (i == 1 and j == -1)
            if a or b or c:
                continue

            evaluation = eval_cell(row_ax + i, col_ax + j, grid, player, opponent)
            if evaluation == 1:
                enemy += 1
                if enemy > 1:
                    return False

            if evaluation == -1:
                return False

    return enemy == 1


def eval_cell(
    row_ax: int, col_ax: int, grid: Grid.Grid, player: Grid.Player, opponent: Grid.Player
) -> int:
    """Evaluate a single cell and return its status."""
    size = (len(grid) - 1) // 2

    if (row_ax < -size) or (col_ax < -size) or (row_ax > size) or (col_ax > size):
        return 0

    coor_grid = Grid.axial_to_grid(row_ax, col_ax, size)
    if grid[coor_grid[0]][coor_grid[1]] == player:
        return -1
    if grid[coor_grid[0]][coor_grid[1]] == opponent:
        return 1
    return 0


def is_final(grid: Grid.Grid, current_player: Grid.Player) -> bool:
    """Check if the game has reached a final state."""
    return len(legals(grid, current_player)) == 0


def score(state: Grid.State, current_player: Grid.Player) -> float:
    """Calculate the score for the given state and player."""
    grid = Grid.state_to_grid(state)
    if current_player == Grid.RED and is_final(grid, current_player):
        return -1
    if current_player == Grid.BLUE and is_final(grid, current_player):
        return 1
    return 0


def play(
    state: Grid.State, action: Grid.ActionGopher, current_player: Grid.Player
) -> Grid.State:
    """Apply an action to the current state and return the new state."""
    grid = Grid.state_to_grid(state)
    size = (len(grid) - 1) // 2
    row, col = action
    coor_grid = Grid.axial_to_grid(row, col, size)
    grid[coor_grid[0]][coor_grid[1]] = current_player
    return Grid.grid_to_state(grid)

def first_move(grid: Grid.Grid) -> List[Tuple[int, int]]:
    """Determine the first move for the player."""
    actions = []
    size = (len(grid) - 1) // 2
    for row in range(-size, size + 1):
        for col in range(-size, size + 1):
            coor_grid = Grid.axial_to_grid(row, col, size)
            if grid[coor_grid[0]][coor_grid[1]] == 0:
                actions.append((row, col))
    return actions


def first_play(grid: Grid.Grid, _: Grid.Player) -> Tuple[float, Tuple[int, int]]:
    """Handle the first play action."""
    selection = False
    actions = first_move(grid)
    for action in actions:
        print(action)
        chrono_start = time.time()
    while not selection:
        row = int(input("Select a row: "))
        col = int(input("Select a col: "))
        for action in actions:
            if (row, col) == action:
                selection = True
                chrono_end = time.time()
                return chrono_end - chrono_start, action
    return chrono_end - chrono_start


def eval_cells_around_opti_legals(
    grid: Grid.Grid, player: Grid.Player, last_action: Union[Grid.ActionGopher, None]
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """Evaluate cells around the last action to optimize legal moves."""
    opponent = player % 2 + 1
    playable = []
    unplayable = []

    if last_action is None:
        return playable, unplayable

    for i in range(-1, 2):
        for j in range(-1, 2):
            a = (i == 0 and j == 0)
            b = (i == -1 and j == 1)
            c = (i == 1 and j == -1)
            if a or b or c:
                continue
            if eval_cells_around(
                last_action[0] + i, last_action[1] + j, grid, player, opponent
            ):
                playable.append((last_action[0] + i, last_action[1] + j))
            else:
                unplayable.append((last_action[0] + i, last_action[1] + j))
    return playable, unplayable


def evaluate(grid: Grid.Grid, player: Grid.Player) -> float:
    """Evaluate the current game state for the player."""
    opponent = player % 2 + 1
    player_legal_moves = len(legals(grid, player))
    opponent_legal_moves = len(legals(grid, opponent))
    
    
    return -score(Grid.grid_to_state(grid),player)*100+player_legal_moves - opponent_legal_moves


@memoize_old
def nega_max(state: Grid.State, player: Grid.Player, depth: int):
    """Negamax algorithm for game state evaluation."""
    grid = Grid.state_to_grid(state)
    if depth == 0 or is_final(grid, player):
        return evaluate(grid, player)

    score_tmp = float("-inf")
    moves = legals(grid, player)

    for move in moves:
        new_state = play(state, move, player)
        cur = -nega_max(new_state, player % 2 + 1, depth - 1)
        score_tmp = max(score_tmp, cur)

    return score_tmp


def nega_max_action(
    state: Grid.State, player: Grid.Player, depth: int
) -> Tuple[float, Grid.ActionGopher]:
    """Get the best action using the Negamax algorithm."""
    best_score = float("-inf")
    best_action = None

    list_actions = legals(Grid.state_to_grid(state), player)
    for possible_action in list_actions:
        new_state = play(state, possible_action, player)
        eval_score = -nega_max(new_state, player % 2 + 1, depth - 1)
        if eval_score > best_score:
            best_score = eval_score
            best_action = possible_action

    return best_score, best_action


def strategy_nega_max(
    env: Grid.Environment,
    state: Grid.State,
    player: Grid.Player,
    _: Grid.Time,
    depth: int = 3,
) -> Tuple[Grid.Environment, Grid.ActionGopher]:
    """Negamax strategy for the player."""
    _, best_action = nega_max_action(state, player, depth)
    return env, best_action


#@memoize
def nega_max_alpha_beta(
    state: Grid.State, player: Grid.Player, depth: int, alpha: float, beta: float
) -> float:
    """Negamax with alpha-beta pruning for game state evaluation."""
    grid = Grid.state_to_grid(state)
    if depth == 0 or is_final(grid, player):
        return evaluate(grid, player)

    score_tmp = float("-inf")
    moves = legals(grid, player)

    for move in moves:
        new_state = play(state, move, player)
        cur = -nega_max_alpha_beta(new_state, player % 2 + 1, depth - 1, -beta, -alpha)
        score_tmp = max(score_tmp, cur)
        alpha = max(alpha, cur)
        if alpha >= beta:
            break

    return score_tmp


def nega_max_action_alpha_beta(
    state: Grid.State, player: Grid.Player, depth: int
) -> Tuple[float, Grid.ActionGopher]:
    """Get the best action using Negamax with alpha-beta pruning."""
    best_score = float("-inf")
    best_action = None
    alpha = float("-inf")
    beta = float("inf")

    list_actions = legals(Grid.state_to_grid(state), player)
    for possible_action in list_actions:
        new_state = play(state, possible_action, player)
        eval_score = -nega_max_alpha_beta(
            new_state, player % 2 + 1, depth - 1, -beta, -alpha  
        )
        if eval_score > best_score:
            best_score = eval_score
            best_action = possible_action
        alpha = max(alpha, eval_score)  

    return best_score, best_action


def strategy_nega_max_alpha_beta(
    env: Grid.Environment,
    state: Grid.State,
    player: Grid.Player,
    _: Grid.Time,
    depth: int = 4,
) -> Tuple[Grid.Environment, Grid.ActionGopher]:
    """Negamax strategy with alpha-beta pruning for the player."""
    best_action=state[0][0]
    for i in state :
        if i[1]!=0:
            _, best_action = nega_max_action_alpha_beta(state, player, depth)
    return env, best_action


def strategy_random(
    env: Grid.Environment, state: Grid.State, player: Grid.Player, _: Grid.Time
) -> Tuple[Grid.Environment, Grid.ActionGopher]:
    """Random strategy for the player."""
    return env, random.choice(legals(Grid.state_to_grid(state), player))

def demander_coup(list_actions,state,player):
    api_key = "Your secret key"
    client = OpenAI(api_key=api_key)
    rules ='''Here are the rules : INTRODUCTION
Gopher is a two player game played on an initially empty, size 6 (or 8...)*
hexagonal grid. The two players, Red and Blue, place their own stones on the
board, one stone per turn. Players are not allowed to pass. Mark Steere designed
Gopher in March, 2021.
PLACEMENTS
All placements are to unoccupied cells. A FRIENDLY CONNECTION is an
adjacency between like colored stones. An ENEMY CONNECTION is an
adjacency between different colored stones.
Red begins the game by placing a stone anywhere on the board. Then, starting
with Blue, players take turns placing a stone which forms exactly one enemy
connection and no friendly connections. See Figure 1.
OBJECT OF THE GAME
The last player to place a stone wins.'''
    try:
        response = client.chat.completions.create(model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant with extensive knowledge of gopher. "},
            {"role": "user", "content": f'''Given the following position, you are player {player} on the board {state}, what move should I play next, here are the legals plays :{list_actions}.? Give only your choice like this format, without any additional text, or else it will break my game: (x,y)'''}
        ])
        return response.choices[0].message.content
    except openai.RateLimitError as e:
        print("Rate limit exceeded. Waiting to retry...")
        time.sleep(60)  # Wait for a minute before retrying
        return demander_coup()
    except openai.OpenAIError as e:
        print(f"An error occurred: {e}")
        return None

def startegy_chatgpt(
    env: Grid.Environment, state: Grid.State, player: Grid.Player, _: Grid.Time
)-> Tuple[Grid.Environment, Grid.ActionGopher]:
    """Playing using openai API for chatgpt"""
    list_actions=legals(Grid.state_to_grid(state),player)
    choix=demander_coup(list_actions,state,player)
    choix=choix.replace("(","").replace(")","").split(",")
    choix=(int(choix[0]),int(choix[1]))
    for i in choix : print(i)
    print(choix,type(choix))
    print(list_actions)


    return env, choix


def game_play(size: int, strategy_red: Grid.Strategy, strategy_blue: Grid.Strategy):
    """Simulate a game play between two strategies."""
    real_size = size - 1
    environment_red = []
    environment_blue = []
    current_player = Grid.RED
    state = Grid.grid_to_state(init_grid(real_size))

    environment_red, action = strategy_red(environment_red, state, current_player, time)
    state = play(state, action, current_player)
    current_player = Grid.BLUE

    while not is_final(Grid.state_to_grid(state), current_player):
        if current_player == Grid.RED:
            environment_red, action = strategy_red(
                environment_red, state, current_player, time
            )
            state = play(state, action, current_player)
            current_player = Grid.BLUE
        else:
            environment_blue, action = strategy_blue(
                environment_blue, state, current_player, time
            )
            state = play(state, action, current_player)
            current_player = Grid.RED

    return score(state, current_player)


def main():
    """Main function to run the game simulations."""
    start = time.time()
    list_scores = []
    for _ in range(1):
        game_score = game_play(6, strategy_nega_max_alpha_beta, strategy_random)
        list_scores.append(game_score)
        print(game_score)
    for game_score in list_scores:
        print(game_score,"c'est le gagnant", end=" ", )
    print()
    print(time.time() - start)

    print(list_scores.count(1), "/", len(list_scores))


if __name__ == "__main__":
    main()
