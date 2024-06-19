"""
Module for Gopher game grid utilities and functions.
"""

from typing import Callable, List, Tuple, Union
import numpy as np

# Types de base utilisés par l'arbitre
Environment = ...  # Ensemble des données utiles (cache, état de jeu...)
Cell = Tuple[int, int]
ActionGopher = Cell
ActionDodo = Tuple[Cell, Cell]  # case de départ -> case d'arrivée
Action = Union[ActionGopher, ActionDodo]
Player = int  # 1 ou 2
State = List[Tuple[Cell, Player]]  # État du jeu pour la boucle de jeu
Score = int
Time = int

Grid = List[List[int]]
Strategy = Callable[[Grid, Player], Action]

RED = 1  # Player 1
DRAW = 0
BLUE = 2  # Player 2
Color = Union[RED, BLUE]

grid_test: Grid = [
    [None, None, None, 1, 2, 3, 4],
    [None, None, 5, 6, 7, 8, 9],
    [None, 10, 11, 12, 13, 14, 15],
    [16, 17, 18, 19, 20, 21, 22],
    [23, 24, 25, 26, 27, 28, None],
    [29, 30, 31, 32, 33, None, None],
    [34, 35, 36, 37, None, None, None]
]

grid_dodo: Grid = [
    [None, None, None, -1, -1, -1, -1],
    [None, None, 0, -1, -1, -1, -1],
    [None, 0, 0, 0, -1, -1, -1],
    [1, 1, 0, 0, 0, -1, -1],
    [1, 1, 1, 0, 0, 0, None],
    [1, 1, 1, 1, 0, None, None],
    [1, 1, 1, 1, None, None, None]
]

data = [
    [None, None, 2, 0, 0],
    [None, 0, 2, 0, 0],
    [1, 1, 2, 1, 1],
    [0, 0, 2, 0, None],
    [0, 0, 2, None, None]
]

###Print functions
def print_grid_flat(grid: Grid):
    """Print grid with flat top."""
    for row in grid:
        count_spaces = 0
        list_elements = []

        for element in row:
            if element is None:
                count_spaces += 1
            elif element == RED:
                list_elements.append("R")
            elif element == BLUE:
                list_elements.append("B")
            else:
                list_elements.append(element)

        for _ in range(count_spaces):
            print(" ", end=" ")

        for e in list_elements:
            print(e, end="   ")

        print("\n")


def print_grid_pointy(grid: Grid):
    """Print grid with pointy top."""
    l = []
    size = (len(grid) + 1) // 2
    for i in range(2 * (size - 1), -(2 * size) + 1, -1):
        a = []
        for j in range(-size + 1, size):
            for k in range(-size + 1, size):
                if j + k == i:
                    a.append([j, k])
        l.append(a)

    tab = []
    for i in l:
        f = []
        for j in i:
            a, b = axial_to_grid(j[0], j[1], size - 1)
            if grid[a][b] is not None:
                if grid[a][b] == RED:
                    f.append("R")
                elif grid[a][b] == BLUE:
                    f.append("B")
                else:
                    f.append(grid[a][b])
        tab.append(f)

    max_width = max(len(row) for row in tab)

    for i, row in enumerate(tab):
        for j, val in enumerate(row):
            tab[i][j] = str(val)

    outp = []
    for i, row in enumerate(tab):
        if len(row) != 0:
            outp.append("    " * (max_width - len(row)) + "       ".join(row))
    for i in outp:
        print(i)

###State to tuple
def state_to_tuple(state: State) -> tuple:
    """Convert state to tuple."""
    return tuple(sorted((tuple(cell), player) for cell, player in state))


# Symmétries
def horizontal_flip_symmetry(state: State) -> State:
    """Apply horizontal flip symmetry to the state."""
    grid = state_to_grid(state)
    symmetric_grid = size_to_2d_array((len(grid) - 1) // 2)
    for i in enumerate(grid):
        for j in enumerate(grid[i]):
            symmetric_grid[i][j] = grid[len(grid) - 1 - i][len(grid) - 1 - j]
    return grid_to_state(symmetric_grid)


def horizontal_symmetry(state: State) -> State:
    """Apply horizontal symmetry to the state."""
    grid = state_to_grid(state)
    symmetric_grid = np.transpose(grid)
    return grid_to_state(symmetric_grid)


def vertical_symmetry(state: State) -> State:
    """Apply vertical symmetry to the state."""
    return horizontal_flip_symmetry(horizontal_symmetry(state))


def rota_60(state: State) -> State:
    """Rotate the state by 60 degrees."""
    outp = []
    for i in state:
        outp.append([(i[0][1], -i[0][0] + i[0][1]), i[1]])
    return outp


def rota_120(state: State) -> State:
    """Rotate the state by 120 degrees."""
    outp = []
    for i in state:
        outp.append([(i[0][1] - i[0][0], -i[0][0]), i[1]])
    return outp


###Conversion functions
def size_to_2d_array(size) -> Grid:
    """Create grid of given size."""
    if size <= 0:
        return None
    liste = []
    for i in range(size * 2 + 1):
        row = []
        for j in range(size * 2 + 1):
            if i + j < size:
                row.append(None)
            elif i + j > 3 * size:
                row.append(None)
            else:
                row.append(0)
        liste.append(row)
    return liste


def axial_to_grid(row, col, size: int) -> Tuple[int, int]:
    """Convert axial coordinates to grid coordinates."""
    return size - col, size + row


def grid_to_axial(row, col, size: int) -> Tuple[int, int]:
    """Convert grid coordinates to axial coordinates."""
    return col - size, -row + size


def state_to_grid(state: State) -> Grid:
    """Convert state to grid."""
    if state is None or len(state) == 0:
        raise ValueError("State cannot be None or empty")

    size = max(max(coor_ax[0], coor_ax[1]) for coor_ax, _ in state)
    grid = size_to_2d_array(size)

    for element in state:
        coor_ax = element[0]
        a, b = axial_to_grid(coor_ax[0], coor_ax[1], size)

        if a < 0 or a >= len(grid) or b < 0 or b >= len(grid):
            raise IndexError(f"Calculated grid indices out of range: A={a}, B={b}")

        if grid[a] is None or grid[a][b] is None:
            raise ValueError(f"Grid position at A={a}, B={b} is None")

        grid[a][b] = element[1]

    return grid


def grid_to_state(grid: Grid) -> State:
    """Convert grid to state."""
    size = (len(grid) - 1) // 2
    state: State = []
    for row in range(len(grid)):
        for col in range(len(grid[row])):
            if grid[row][col] is not None:
                state.append([grid_to_axial(row, col, size), grid[row][col]])
    return state


def is_grid_empty(grid: Grid) -> bool:
    """Check if the grid is empty."""
    size = (len(grid) - 1) // 2

    for i in range(-size, size + 1):
        for j in range(-size, size + 1):
            coor_grid = axial_to_grid(i, j, size)
            if grid[coor_grid[0]][coor_grid[1]] not in (0, None):
                return False
    return True


###Main function
def main():
    """Main function."""
    print_grid_pointy(grid_test)


if __name__ == "__main__":
    main()
