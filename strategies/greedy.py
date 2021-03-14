from typing import Set, Iterator, Collection, Dict, Any, Optional, List, Callable

from config_loader import StrategyParams
from heuristics import get_heuristic_from_strategy_params
from node import InformedNode
from state import State
from strategy_stats import StrategyStats
import heapq


def greedy(init_state: State, strategy_stats: StrategyStats, strategy_params: StrategyParams) -> Collection[State]:
    heuristic: Callable[[InformedNode], int] = get_heuristic_from_strategy_params(strategy_params)

    root: InformedNode = InformedNode(init_state, None, heuristic)

    visited_states: Set[State] = set()
    visited_states.add(root.state)

    priority_queue: List[InformedNode] = [root]
    heapq.heapify(priority_queue)
    strategy_stats.inc_leaf_node_count()

    while priority_queue:

        current_node: InformedNode = heapq.heappop(priority_queue)

        if current_node.has_won():
            return current_node.get_state_list()

        new_nodes_iter: Iterator[InformedNode] = filter(lambda node: node.state not in visited_states,
                                                        current_node.expand())
        has_children: bool = False

        for node in new_nodes_iter:
            visited_states.add(node.state)
            heapq.heappush(priority_queue, node)
            strategy_stats.inc_leaf_node_count()
            has_children = True

        if has_children:
            strategy_stats.dec_leaf_node_count()

        strategy_stats.inc_exploded_node_count()

    return [init_state]
