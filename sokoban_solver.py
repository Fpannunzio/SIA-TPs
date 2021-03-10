from time import perf_counter
from typing import List, Callable, Dict, Iterable, Collection

from strategies.dfs import dfs
from strategies.bfs import bfs
from strategies.iddfs import iddfs
from strategies.iddfs_dup_states import iddfs_dup
from strategy_stats import StrategyStats
from visualization.game_renderer import GameRenderer
import sys

from state import State
from _level_loader import load_initial_state

# Declare available strategies
strategy_map: Dict[str, Callable[[State, StrategyStats], Collection[State]]] = {
    'DFS': dfs,
    'BFS': bfs,
    'IDDFS': iddfs,
    'IDDFS_DUP': iddfs_dup,
    # 'GREEDY': greedy,
    # 'A*': a_star,
}


def main(level_name: str, strategy_name: str, render: bool = True):

    # Load initial state from level file selected
    initial_state: State = load_initial_state(level_name)

    # Create selected strategy stats holder
    strategy_stats: StrategyStats = StrategyStats(strategy_name, level_name)

    # Solve Sokoban using selected strategy
    states: Collection[State] = solve_sokoban(strategy_name, initial_state, strategy_stats)

    # Print selected strategy stats
    strategy_stats.print_stats()

    # Render Solution
    if render:
        GameRenderer(states).render()


def solve_sokoban(strategy_name: str, init_state: State, strategy_stats: StrategyStats) -> Collection[State]:

    if strategy_name not in strategy_map:
        raise RuntimeError(f'Invalid strategy {strategy_name}. Currently supported: {strategy_map.keys()}')

    start: float = perf_counter()
    states: Collection[State] = strategy_map[strategy_name](init_state, strategy_stats)
    end: float = perf_counter()

    strategy_stats.set_runtime(start, end)
    strategy_stats.set_solution_move_count(len(states))

    return states


# Usage: python3 sokoban_solver.py [OPTIONS] [level_name] [solve_strategy]
if __name__ == "__main__":
    argv = sys.argv

    # Handle option --no-render
    render: bool = False
    try:
        argv.remove('--no-render')
    except ValueError:
        render = True

    level_name_arg: str = (argv[1] if len(argv) >= 2 else "level.txt")
    strategy_arg: str = (argv[2] if len(argv) >= 3 else "BFS")

    main(level_name_arg, strategy_arg, render)
