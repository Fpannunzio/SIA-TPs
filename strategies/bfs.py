from collections import deque
from typing import Deque, Set, Iterator, Collection

from config_loader import StrategyParams
from node import Node
from state import State
from strategy_stats import StrategyStats


def bfs(init_state: State, strategy_stats: StrategyStats, strategy_params: StrategyParams) -> Collection[State]:

    filter_lost_states: bool = (strategy_params.get('filter_lost_states', True) if strategy_params else True)
    root = Node(init_state, None)

    visited_states: Set[State] = set()
    visited_states.add(root.state)

    queue: Deque[Node] = deque()
    queue.append(root)
    strategy_stats.inc_leaf_node_count()

    while queue:
        current_node: Node = queue.popleft()

        if current_node.has_won():
            return current_node.get_state_list()

        new_nodes: Iterator[Node] = filter(lambda node: node.state not in visited_states, current_node.expand(filter_lost_states))
        has_children: bool = False

        for node in new_nodes:
            visited_states.add(node.state)
            queue.append(node)
            strategy_stats.inc_leaf_node_count()
            has_children = True

        if has_children:
            strategy_stats.dec_leaf_node_count()

        strategy_stats.inc_exploded_node_count()

    return []


