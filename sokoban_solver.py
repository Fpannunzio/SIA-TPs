from time import perf_counter
from typing import Callable, Dict, Collection, Any, Optional

from config_loader import Config
from strategies.dfs import dfs
from strategies.bfs import bfs
from strategies.iddfs import iddfs
from strategies.iddfs_dup_states import iddfs_dup
from strategies.greedy import greedy
from strategy_stats import StrategyStats
from visualization.game_renderer import GameRenderer
import sys

from state import State
from _level_loader import load_initial_state

# Declare available strategies
strategy_map: Dict[str, Callable[[State, StrategyStats, Optional[Dict[str, Any]]], Collection[State]]] = {
    'DFS': dfs,
    'BFS': bfs,
    'IDDFS': iddfs,
    'IDDFS_DUP': iddfs_dup,
    'GREEDY': greedy,
    # 'A*': a_star,
}


def main(config_file: str):

    # Load Config from config.yaml
    config: Config = Config(config_file)

    # Load initial state from level file selected
    initial_state: State = load_initial_state(config.level)

    # Create selected strategy stats holder
    strategy_stats: StrategyStats = StrategyStats(config.strategy, config.level)

    # Solve Sokoban using selected strategy
    states: Collection[State] = solve_sokoban(config.strategy, initial_state, strategy_stats, config.strategy_params)

    # Print selected strategy stats
    strategy_stats.print_stats()

    # Render Solution
    if config.render:
        GameRenderer(states).render()


def solve_sokoban(strategy_name: str, init_state: State, strategy_stats: StrategyStats,
                  strategy_params: Optional[Dict[str, Any]]) -> Collection[State]:

    if strategy_name not in strategy_map:
        raise RuntimeError(f'Invalid strategy {strategy_name}. Currently supported: {strategy_map.keys()}')

    start: float = perf_counter()
    states: Collection[State] = strategy_map[strategy_name](init_state, strategy_stats, strategy_params)
    end: float = perf_counter()

    strategy_stats.set_runtime(start, end)
    strategy_stats.set_solution_move_count(len(states))

    return states


# Usage: python3 sokoban_solver.py [config_file]
if __name__ == "__main__":
    argv = sys.argv

    config_file: str = 'config.yaml'

    if len(argv) > 1:
        config_file = argv[1]

    main(config_file)
