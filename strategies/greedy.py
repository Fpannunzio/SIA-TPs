from typing import Set, Iterator, Collection,Dict, Any, Optional

from node import InformedNode
from state import State
from strategy_stats import StrategyStats
import heapq


def manhattan_distance(current_node: InformedNode) -> int:
    heuristic = 0
    for target in current_node.state.level_map.targets:
        distances: Set[int] = set()
        for box in current_node.state.boxes:
            distances.add(abs(target.x - box.x) + abs(target.y - box.y))
        heuristic += min(distances)

    return heuristic


def greedy(init_state: State, strategy_stats: StrategyStats, strategy_params: Optional[Dict[str, Any]]) -> Collection[State]:

    filter_lost_states: bool = (strategy_params.get('filter_lost_states', True) if strategy_params else True)
    root: InformedNode = InformedNode(init_state, None, manhattan_distance)

    visited_states: Set[State] = set()
    visited_states.add(root.state)

    priority_queue = [root]
    heapq.heapify(priority_queue)

    while priority_queue:

        current_node: InformedNode = heapq.heappop(priority_queue)

        if current_node.has_won():
            return current_node.get_state_list()

        new_nodes_iter: Iterator[InformedNode] = filter(lambda node: node.state not in visited_states, current_node.expand(filter_lost_states))

        for node in new_nodes_iter:
            visited_states.add(node.state)
            heapq.heappush(priority_queue, node)

        # Update Strategy Stats
        strategy_stats.inc_exploded_node_count()
        strategy_stats.set_current_nodes_stored(len(priority_queue))

    return [init_state]