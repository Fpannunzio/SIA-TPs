from typing import List, Tuple, Optional

from state import State
from game_engine import process_move

class StateWrapper:
    def __init__(self, current_state: State):
        self.current_state: State = current_state

    def set_previous(self, previous_state):
        self.previous_state = previous_state

def dfs(init_state: State) -> List[State]:
    
    if init_state is None:
        return None
    
    first_wrapper = StateWrapper(init_state)
    first_wrapper.set_previous(None)
    list = [first_wrapper]
    stack = [first_wrapper]

    directions: List[Tuple[int, int]] = [(0, 1), (0, -1), (1, 0), (-1, 0)] 

    while stack:
        state = stack.pop()

        if state.current_state.has_won():
            final_list = []
            solution(final_list, state)
            return final_list

        # print(state.current_state.player_pos, state.current_state.boxes, state.current_state.targets_remaining)
        for direction in directions:
            wrapper_state = StateWrapper(process_move(state.current_state, direction))
            wrapper_state.set_previous(state)
            
            if not wrapper_state.current_state is None and state_not_visited(list, wrapper_state.current_state):
                list.append(wrapper_state)
                stack.append(wrapper_state)
    return [init_state]

def state_not_visited(states_list: List[StateWrapper], aux_state: State) -> bool:
    for state in states_list:
        if (state.current_state == aux_state):
            return False
    return True

def solution(final_list: List[State], last: StateWrapper):
    if last.previous_state is None:
        return
    solution(final_list, last.previous_state)
    final_list.append(last.current_state)