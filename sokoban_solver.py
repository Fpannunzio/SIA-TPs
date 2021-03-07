from typing import List, Callable

from dfs import dfs
from bfs import bfs
from visualization.game_renderer import GameRenderer
import sys

from state import State
from _level_loader import load_initial_state


# TODO: Add __repr__ to everything
def main(level_name: str, strategy: str):

    # Load initial state from level file selected
    initial_state: State = load_initial_state(level_name)

    # Solve Sokoban using selected strategy
    states: List[State] = solve_sokoban(strategy, initial_state)

    print(f'Solution found in {len(states)} steps')

    # Render Solution
    GameRenderer(states).render()


# TODO: Replace with dictionary
def solve_sokoban(strategy_name: str, init_state: State) -> List[State]:
    strategy: Callable[[State], List[State]]

    if strategy_name == 'DFS':
        strategy = dfs

    elif strategy_name == 'BFS':
        strategy = bfs

    # elif strategy_name == 'IDDFS':
    #     pass  # TODO: iddfs(init_state)

    else:
        raise RuntimeError(f'Invalid strategy {strategy_name}. Currently supported: [BFS, DFS, IDDFS]')

    return strategy(init_state)


# Usage: python3 Sokoban [level_name] [solve_strategy]
if __name__ == "__main__":
    argv = sys.argv

    level_name_arg: str = (argv[1] if len(argv) >= 2 else "level.txt")
    strategy_arg: str = (argv[2] if len(argv) >= 3 else "BFS")

    main(level_name_arg, strategy_arg)
