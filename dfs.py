from collections import deque
from typing import List, Deque, Set

from node import Node
from state import State


def dfs(init_state: State) -> List[State]:
    
    root = Node(init_state, None)

    visited_states: Set[State] = set()
    visited_states.add(root.state)

    stack: Deque[Node] = deque()
    stack.append(root)

    while stack:
        current_node: Node = stack.pop()

        if current_node.has_won():
            # TODO: print meaningfull information such as depth, exploded nodes, time, etc
            return current_node.get_state_list()

        new_nodes: List[Node] = list(filter(lambda node: node.state not in visited_states, current_node.expand()))

        for node in new_nodes:
            visited_states.add(node.state)
            stack.append(node)

    return [init_state]
