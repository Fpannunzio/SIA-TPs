from collections import deque
from typing import Deque, Set, Iterator, Collection, Dict, Any

from node import Node
from state import State
from strategy_stats import StrategyStats


def bfs(init_state: State, strategy_stats: StrategyStats, strategy_params: Dict[str, Any]) -> Collection[State]:
    root = Node(init_state, None)

    visited_states: Set[State] = set()
    visited_states.add(root.state)

    queue: Deque[Node] = deque()
    queue.append(root)

    while queue:
        current_node: Node = queue.popleft()

        if current_node.has_won():
            return current_node.get_state_list()

        new_nodes: Iterator[Node] = filter(lambda node: node.state not in visited_states, current_node.expand())

        for node in new_nodes:
            visited_states.add(node.state)
            queue.append(node)

        # Update Strategy Stats
        strategy_stats.inc_exploded_node_count()
        strategy_stats.set_current_nodes_stored(len(queue))

    return [init_state]


