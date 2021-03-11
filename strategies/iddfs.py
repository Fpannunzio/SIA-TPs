from collections import deque
from typing import Deque, Set, Iterator, Collection, Tuple, Dict, Any

from node import Node
from state import State
from strategy_stats import StrategyStats


# Sacar del set de ancestros todos aquellos estados que no forman parte del camino al estado actual.
# Teniendo en cuenta que se recorre el arbol de forma prefija, basta con analizar la profundidad
# de los nodos para saber si pueden ser sacados.
def purge_parent_stack(depth: int, parent_nodes: Deque[Node], ancestors: Set[State]):
    if parent_nodes:
        while parent_nodes[-1].depth >= depth:
            parent_to_be_removed: Node = parent_nodes.pop()
            ancestors.remove(parent_to_be_removed.state)


def iddfs(init_state: State, strategy_stats: StrategyStats, strategy_params: Dict[str, Any]) -> Collection[State]:
    if not strategy_params or 'step' not in strategy_params:
        step: int = 10  # default step
    else:
        step = strategy_params['step']

    root: Node = Node(init_state, None)

    stack: Deque[Node] = deque()

    # Nodos frontera que fueron cortados por limite de profundidad y van a ser utilziados
    # para empezar a busacar cuando se investigue la profundidad siguiente. Ademas, tienen un set
    # con todos los estados anteriores asi mismo, utilizado para evitar repetir estados y caer en ciclos
    edge_nodes: Deque[Tuple[Node, Set[State]]] = deque()
    edge_nodes.append((root, set()))

    while edge_nodes:

        edge_node: Node
        ancestors: Set[State]

        (edge_node, ancestors) = edge_nodes.popleft()

        # Profundidad aumenta en funcion de un paso fijo.
        max_depth: int = edge_node.depth + step

        parent_nodes: Deque[Node] = deque()

        stack.append(edge_node)

        while stack:

            current_node: Node = stack.pop()

            if current_node.depth <= max_depth:

                if current_node.has_won():
                    return current_node.get_state_list()

                purge_parent_stack(current_node.depth, parent_nodes, ancestors)

                new_nodes_iter: Iterator[Node] = filter(lambda node: node.state not in ancestors, current_node.expand())

                has_children: bool = False

                for new_node in new_nodes_iter:
                    has_children = True
                    stack.append(new_node)

                if has_children:
                    parent_nodes.append(current_node)
                    ancestors.add(current_node.state)
                    strategy_stats.inc_exploded_node_count()

            else:
                edge_nodes.append((current_node, ancestors.copy()))

            strategy_stats.set_current_nodes_stored(len(stack))


    return [init_state]
