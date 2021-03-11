from collections import deque
from functools import total_ordering
from typing import List, Iterator, Optional, Deque, Collection, Callable

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

    def expand(self, filter_lost_states: bool) -> Iterator['Node']:
        expanded_states_iter: Iterator[State] = map(self.state.move_player, self.state.get_valid_player_moves())

        if filter_lost_states:
            expanded_states_iter = filter(lambda state: not state.has_lost(), expanded_states_iter)

        return map(lambda state: Node(state, self), expanded_states_iter)

    def __repr__(self) -> str:
        return f'Node(state={repr(self.state)}, parent_id={id(self.parent)})'


class InformedNode(Node):

    def __init__(self, state: State, parent: Optional['InformedNode'], heuristic_func: Callable[['InformedNode'], int]):
        super().__init__(state, parent)
        self.heuristic_func = heuristic_func
        self.heuristic_val: int = self.heuristic_func(self)

    def expand(self, filter_lost_states: bool = True) -> Iterator['InformedNode']:
        expanded_states_iter: Iterator[State] = map(self.state.move_player, self.state.get_valid_player_moves())

        winnable_states_iter = filter(lambda state: not state.has_lost(), expanded_states_iter)

        return map(lambda state: InformedNode(state, self, self.heuristic_func), winnable_states_iter)

    # This class is design to be ordered only around the heuristic definition provided, independent of the state
    def __eq__(self, o: object) -> bool:
        if not isinstance(o, InformedNode):
            return False

        return self.heuristic_val == o.heuristic_val

    def __hash__(self) -> int:
        return hash(self.heuristic_val)

    @total_ordering
    def __lt__(self, other: 'InformedNode') -> bool:
        return self.heuristic_val < other.heuristic_val
