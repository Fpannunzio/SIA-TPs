from collections import deque
from typing import List, Iterator, Optional, Deque, Collection

from state import State


class Node:

    def __init__(self, state: State, parent: Optional['Node']):
        self.state: State = state
        self.parent: Optional[Node] = parent
        self.depth: int = (parent.depth + 1 if parent else 0)

    def has_won(self) -> bool:
        return self.state.has_won()

    def get_state_list(self) -> Collection[State]:
        state_queue: Deque[State] = deque()
        current_node: Optional[Node] = self

        while current_node:
            state_queue.appendleft(current_node.state)
            current_node = current_node.parent

        return state_queue

    def expand(self) -> Iterator['Node']:
        expanded_states_iter: Iterator[State] = map(self.state.move_player, self.state.get_valid_player_moves())

        winnable_states_iter: Iterator[State] = filter(lambda state: not state.has_lost(), expanded_states_iter)

        return map(lambda state: Node(state, self), winnable_states_iter)

    def _build_state_list_rec(self, state_list: List[State]) -> None:
        if self.parent:
            self.parent._build_state_list_rec(state_list)

        state_list.append(self.state)

    def __repr__(self) -> str:
        return f'Node(state={repr(self.state)}, parent_id={id(self.parent)})'
