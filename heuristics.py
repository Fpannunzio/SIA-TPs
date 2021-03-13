from typing import Callable, Dict

from config_loader import StrategyParams
from node import InformedNode
import math

# TODO: Agregar dos heuristicas mas y definir los nombres de todas


# Minimiza la suma de las distancias entre un target y la caja mas cercana.
# Busca que t.odo target tenga una caja cerca
# La distancia usada es la Distancia Manhattan
# Es admisible pues se asume que no hay paredes, es decir, el mejor caso posible
def target_near_box_heuristic(current_node: InformedNode) -> int:
    total_distance: int = 0
    for target in current_node.state.level_map.targets:
        total_distance += min(abs(target.x - box.x) + abs(target.y - box.y) for box in current_node.state.boxes)

    return total_distance


def euclidean_distance_heuristic(current_node: InformedNode) -> int:
    total_distance: int = 0
    for target in current_node.state.level_map.targets:
        total_distance += min(math.sqrt((target.x - box.x)**2 + (target.y - box.y)**2) for box in current_node.state.boxes)

    return total_distance


def open_goal_heuristic(current_node: InformedNode) -> int:
    return current_node.state.targets_remaining


heuristic_map: Dict[str, Callable[[InformedNode], int]] = {
    'target_near_box': target_near_box_heuristic,
    'euclidean_distance': euclidean_distance_heuristic,
    'open_goal': open_goal_heuristic
}


def get_heuristic(heuristic_name: str) -> Callable[[InformedNode], int]:
    if heuristic_name not in heuristic_map:
        raise ValueError(f'Invalid heuristic {heuristic_name}. Currently supported: {heuristic_map.keys()}')

    return heuristic_map[heuristic_name]


def get_heuristic_from_strategy_params(strategy_params: StrategyParams) -> Callable[[InformedNode], int]:
    if not strategy_params or 'heuristic' not in strategy_params:
        raise ValueError(f'A heuristic was not provided. For any informed strategy, include heuristic name in config'
                         f' (strategy: params: heuristic: heuristic_name)')

    return get_heuristic(strategy_params['heuristic'])
