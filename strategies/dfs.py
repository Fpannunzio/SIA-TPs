from collections import deque
from typing import List, Deque, Set, Iterator, Collection, Dict

from node import Node
from state import State
from strategy_stats import StrategyStats


def dfs(init_state: State, strategy_stats: StrategyStats, strategy_params: Dict[str, str]) -> Collection[State]:
    
    root = Node(init_state, None)

    visited_states: Set[State] = set()
    visited_states.add(root.state)

    stack: Deque[Node] = deque()
    stack.append(root)

    while stack:
        current_node: Node = stack.pop()

        if current_node.has_won():
            return current_node.get_state_list()

        new_nodes_iter: Iterator[Node] = filter(lambda node: node.state not in visited_states, current_node.expand())

        for node in new_nodes_iter:
            visited_states.add(node.state)
            stack.append(node)

        # Update Strategy Stats
        strategy_stats.inc_exploded_node_count()
        strategy_stats.set_current_nodes_stored(len(stack))

    return [init_state]
