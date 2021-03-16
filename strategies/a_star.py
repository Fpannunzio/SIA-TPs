from typing import Set, Iterator, Collection, List, Callable

from config_loader import StrategyParams
from heuristics import get_heuristic_from_strategy_params
from node import InformedNode, CostInformedNode
from state import State
from strategy_stats import StrategyStats
import heapq


def a_star(init_state: State, strategy_stats: StrategyStats, strategy_params: StrategyParams) -> Collection[State]:

    heuristic: Callable[[State], int] = get_heuristic_from_strategy_params(strategy_params)

    root: InformedNode = CostInformedNode(init_state, None, heuristic)

    visited_states: Set[State] = set()
    visited_states.add(root.state)

    priority_queue: List[InformedNode] = [root]
    heapq.heapify(priority_queue)

    while priority_queue:

        current_node: InformedNode = heapq.heappop(priority_queue)

        if current_node.has_won():
            strategy_stats.set_boundary_node_count(len(priority_queue))
            return current_node.get_state_list()

        new_nodes_iter: Iterator[InformedNode] = filter(lambda node: node.state not in visited_states,
                                                        current_node.expand())

        for node in new_nodes_iter:
            visited_states.add(node.state)
            heapq.heappush(priority_queue, node)

        strategy_stats.inc_exploded_node_count()

    return []
