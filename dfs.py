from typing import List

from state import State
from game_engine import game

# Mock de la clase dfs. Devuelve la lista de estados ordenados que hay que seguir para ganar el juego
# Solo muevo el jugador porque paja las cajas
def dfs(init_state: State) -> List[State]:
    list = [init_state]
    aux_state = State
    
    aux_state = game(init_state, (0, 1))
    list.append(aux_state)
    aux_state = game(aux_state, (-1, 0))
    list.append(aux_state)
    aux_state = game(aux_state, (0, -1))
    list.append(aux_state)
    aux_state = game(aux_state, (-1, 0))
    list.append(aux_state)
    aux_state = game(aux_state, (0, -1))
    list.append(aux_state)
    aux_state = game(aux_state, (1, 0))
    list.append(aux_state)
    aux_state = game(aux_state, (1, 0))
    list.append(aux_state)
    print(aux_state.has_won())
    # list.append(game(init_state, (3, 3)))
    # list.append(game(init_state, (3, 4)))
    # list.append(State(init_state.level_map, (4, 3), init_state.boxes))
    # list.append(State(init_state.level_map, (4, 4), init_state.boxes))
    # list.append(State(init_state.level_map, (3, 4), init_state.boxes))
    # list.append(State(init_state.level_map, (3, 3), init_state.boxes))
    

    return list
