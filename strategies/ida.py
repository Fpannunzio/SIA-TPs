from collections import deque
from typing import Set, Iterator, Collection, List, Callable, Deque

from config_loader import StrategyParams
from heuristics import get_heuristic_from_strategy_params
from node import InformedNode, CostInformedNode
from state import State
from strategy_stats import StrategyStats
import heapq


def ida(init_state: State, strategy_stats: StrategyStats, strategy_params: StrategyParams) -> Collection[State]:

    heuristic: Callable[[State], int] = get_heuristic_from_strategy_params(strategy_params)

    root: InformedNode = CostInformedNode(init_state, None, heuristic)

    visited_states: Set[State] = set()
    visited_states.add(root.state)

    edge_nodes: List[InformedNode] = [root]
    heapq.heapify(edge_nodes)

    while edge_nodes:

        limit_node: InformedNode = heapq.heappop(edge_nodes)

        limit: int = limit_node.get_heuristic_val()

        stack: Deque[InformedNode] = deque()
        stack.append(limit_node)

        while stack:
            current_node: InformedNode = stack.pop()

            if current_node.has_won():
                strategy_stats.set_boundary_node_count(len(stack) + len(edge_nodes))
                return current_node.get_state_list()

            if current_node.heuristic_val > limit:
                heapq.heappush(edge_nodes, current_node)
                continue

            new_nodes_iter: Iterator[InformedNode] = filter(lambda node: node.state not in visited_states,
                                                            current_node.expand())

            for node in new_nodes_iter:
                visited_states.add(node.state)
                stack.append(node)

            strategy_stats.inc_exploded_node_count()

    return []
