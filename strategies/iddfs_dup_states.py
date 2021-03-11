from collections import deque
from typing import Deque, Set, Iterator, Collection, Dict, Any, Optional

from node import Node
from state import State
from strategy_stats import StrategyStats


def iddfs_dup(init_state: State, strategy_stats: StrategyStats, strategy_params: Optional[Dict[str, Any]]) -> Collection[State]:

    filter_lost_states: bool = (strategy_params.get('filter_lost_states', True) if strategy_params else True)
    step: int = (strategy_params.get('filter_lost_states', 10) if strategy_params else 10)  # Default Step 10

    root: Node = Node(init_state, None)

    stack: Deque[Node] = deque()

    # Nodos frontera que fueron cortados por limite de profundidad y van a ser utilziados
    # para empezar a busacar cuando se investigue la profundidad siguiente. Ademas, tienen un set
    # con todos los estados anteriores asi mismo, utilizado para evitar repetir estados y caer en ciclos
    edge_nodes: Deque[Node] = deque()
    edge_nodes.append(root)

    visited_states: Set[State] = set()
    visited_states.add(root.state)

    while edge_nodes:

        edge_node: Node = edge_nodes.popleft()

        # Profundidad aumenta en funcion de un paso fijo.
        max_depth: int = edge_node.depth + step

        stack.append(edge_node)

        while stack:

            current_node: Node = stack.pop()

            if current_node.depth <= max_depth:

                if current_node.has_won():
                    return current_node.get_state_list()

                new_nodes_iter: Iterator[Node] = filter(lambda node: node.state not in visited_states, current_node.expand(filter_lost_states))

                for new_node in new_nodes_iter:
                    visited_states.add(new_node.state)
                    stack.append(new_node)

                strategy_stats.inc_exploded_node_count()

            else:
                edge_nodes.append(current_node)

        strategy_stats.set_current_nodes_stored(666)

    return [init_state]
