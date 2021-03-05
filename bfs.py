from typing import List, Tuple, Optional

from state import State
from game_engine import process_move

class StateWrapper:
    def __init__(self, current_state: State):
        self.current_state: State = current_state

    def set_previous(self, previous_state):
        self.previous_state = previous_state

def bfs(init_state: State) -> List[State]:
    
    if init_state is None:
        return None
    
    first_wrapper = StateWrapper(init_state)
    first_wrapper.set_previous(None)
    list = [first_wrapper]
    queue = [first_wrapper]

    directions: List[Tuple[int, int]] = [(0, 1), (0, -1), (1, 0), (-1, 0)]   
    while queue:
        state = queue.pop(0)
        
        if state.current_state.has_won():
            final_list = []
            solution(final_list, state)
            for state in final_list:
                print(state.player)
            return final_list
        
        for direction in directions:
            wrapper_state = StateWrapper(process_move(state.current_state, direction))
            wrapper_state.set_previous(state)

            if not wrapper_state.current_state is None and state_not_visited(list, wrapper_state.current_state):
                list.append(wrapper_state)
                queue.append(wrapper_state)
    
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
    


