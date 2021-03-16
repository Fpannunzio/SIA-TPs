from collections import deque
from typing import Deque, Set, Iterator, Collection, Dict

from config_loader import StrategyParams
from node import Node
from state import State
from strategy_stats import StrategyStats


def iddfs(init_state: State, strategy_stats: StrategyStats, strategy_params: StrategyParams) -> Collection[State]:

    filter_lost_states: bool = (strategy_params.get('filter_lost_states', True) if strategy_params else True)
    step: int = (strategy_params.get('step', 10) if strategy_params else 10)  # Default Step 10

    root: Node = Node(init_state, None)

    stack: Deque[Node] = deque()

    # Nodos frontera que fueron cortados por limite de profundidad y van a ser utilziados
    # para empezar a busacar cuando se investigue la profundidad siguiente.
    edge_nodes: Deque[Node] = deque()
    edge_nodes.append(root)

    visited_states_depth_dict: Dict[State, int] = dict()
    visited_states_depth_dict[root.state] = 0

    while edge_nodes:

        edge_node: Node = edge_nodes.popleft()

        # Profundidad aumenta en funcion de un paso fijo.
        max_depth: int = edge_node.depth + step

        stack.append(edge_node)

        while stack:
            current_node: Node = stack.pop()

            if current_node.depth >= max_depth:
                edge_nodes.append(current_node)
                continue

            if current_node.has_won():
                strategy_stats.set_boundary_node_count(len(stack) + len(edge_nodes))
                return current_node.get_state_list()

            new_nodes_iter: Iterator[Node] = \
                filter(
                    lambda node: node.state not in visited_states_depth_dict.keys() or node.depth < visited_states_depth_dict[node.state],
                    current_node.expand(filter_lost_states)
                )

            for new_node in new_nodes_iter:
                visited_states_depth_dict[new_node.state] = new_node.depth
                stack.append(new_node)

            strategy_stats.inc_exploded_node_count()

    return []
