import itertools
from typing import Callable, Dict, List, Iterable, FrozenSet

from config_loader import StrategyParams
from map import Position
from state import State


def manhattan_distance(pos1: Position, pos2: Position) -> int:
    return abs(pos1.x - pos2.x) + abs(pos1.y - pos2.y)


def manhattan_distance_sum(iter1: Iterable[Position], iter2: Iterable[Position]):
    return sum(manhattan_distance(two_pos[0], two_pos[1]) for two_pos in zip(iter1, iter2))


# Minimiza la suma de las distancias entre un target y la caja mas cercana.
# Busca que t.odo target tenga una caja cerca
# La distancia usada es la Distancia Manhattan
# Es admisible pues se asume que no hay paredes, es decir, el mejor caso posible
def manhattan_distance_target_box_heuristic(state: State) -> int:
    total_distance: int = 0
    for target in state.level_map.targets:
        total_distance += min(manhattan_distance(target, box) for box in state.boxes)

    return total_distance


# For manhattan_distance_unique_target_box_heuristic
unique_target_box_dist_cache: Dict[FrozenSet[Position], int] = {}


def manhattan_distance_unique_target_box_heuristic(state: State) -> int:
    frozen_boxes: FrozenSet[Position] = frozenset(state.boxes)
    if frozen_boxes in unique_target_box_dist_cache:
        return unique_target_box_dist_cache[frozen_boxes]

    targets: List[Position] = list(state.level_map.targets)

    # Distancia imposiblemente grande
    min_dist: int = len(state.level_map.map) * len(state.level_map.map[0])

    for boxes in itertools.permutations(frozen_boxes):
        min_dist = min(min_dist, manhattan_distance_sum(targets, boxes))

    unique_target_box_dist_cache[frozen_boxes] = min_dist

    return min_dist


def manhattan_distance_player_box_heuristic(state: State) -> int:
    player_pos: Position = state.player_pos
    return min(manhattan_distance(player_pos, box) for box in state.boxes)


def open_goal_heuristic(state: State) -> int:
    return state.targets_remaining


def player_box_dist_plus_open_goal_heuristic(state: State) -> int:
    return manhattan_distance_player_box_heuristic(state) + open_goal_heuristic(state)


heuristic_map: Dict[str, Callable[[State], int]] = {
    'target_box_dist': manhattan_distance_target_box_heuristic,
    'player_box_dist': manhattan_distance_player_box_heuristic,
    'open_goal': open_goal_heuristic,
    'player_box_dist_plus_open_goal': player_box_dist_plus_open_goal_heuristic,
    'unique_target_box_dist': manhattan_distance_unique_target_box_heuristic
}


def get_heuristic(heuristic_name: str) -> Callable[[State], int]:
    if heuristic_name not in heuristic_map:
        raise ValueError(f'Invalid heuristic {heuristic_name}. Currently supported: {heuristic_map.keys()}')

    return heuristic_map[heuristic_name]


def get_heuristic_from_strategy_params(strategy_params: StrategyParams) -> Callable[[State], int]:
    if not strategy_params or 'heuristic' not in strategy_params:
        raise ValueError(f'A heuristic was not provided. For any informed strategy, include heuristic name in config'
                         f' (strategy: params: heuristic: heuristic_name)')

    return get_heuristic(strategy_params['heuristic'])
