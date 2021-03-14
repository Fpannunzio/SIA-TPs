from collections import deque
from typing import Set, Iterator, Collection, Dict, Any, Optional, List, Callable, Deque

from config_loader import StrategyParams
from heuristics import get_heuristic_from_strategy_params
from node import InformedNode
from state import State
from strategy_stats import StrategyStats
import heapq


def astar_wrapper(heuristic: Callable[[InformedNode], int]) -> Callable[[InformedNode], int]:
    return lambda node: heuristic(node) + node.depth


def ida(init_state: State, strategy_stats: StrategyStats, strategy_params: StrategyParams) -> Collection[State]:

    heuristic: Callable[[InformedNode], int] = get_heuristic_from_strategy_params(strategy_params)

    root: InformedNode = InformedNode(init_state, None, astar_wrapper(heuristic))

    visited_states: Set[State] = set()
    visited_states.add(root.state)

    edge_nodes: List[InformedNode] = [root]
    strategy_stats.inc_leaf_node_count()

    while edge_nodes:

        heapq.heapify(edge_nodes)

        limit_node: InformedNode = heapq.heappop(edge_nodes)

        limit: int = limit_node.heuristic_val

        stack: Deque[InformedNode] = deque()
        stack.append(limit_node)

        edge_nodes.append(limit_node)

        while stack:

            current_node: InformedNode = stack.pop()

            if current_node.has_won():
                return current_node.get_state_list()

            if current_node.heuristic_val > limit:
                visited_states.remove(current_node.state)
                break

            edge_nodes.remove(current_node)

            new_nodes_iter: Iterator[InformedNode] = filter(lambda node: node.state not in visited_states,
                                                            current_node.expand())
            has_children: bool = False

            for node in new_nodes_iter:
                visited_states.add(node.state)
                stack.append(node)
                edge_nodes.append(node)
                strategy_stats.inc_leaf_node_count()
                has_children = True

            if has_children:
                strategy_stats.dec_leaf_node_count()

            strategy_stats.inc_exploded_node_count()

        while stack:
            visited_states.remove(stack.pop().state)

    return [init_state]
