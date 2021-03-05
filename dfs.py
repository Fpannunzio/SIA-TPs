from typing import List

from state import State


# Mock de la clase dfs. Devuelve la lista de estados ordenados que hay que seguir para ganar el juego
# Solo muevo el jugador porque paja las cajas
def dfs(init_state: State) -> List[State]:
    list = [init_state]

    list.append(State(init_state.level_map, (12, 7), init_state.boxes))
    list.append(State(init_state.level_map, (11, 7), init_state.boxes))
    list.append(State(init_state.level_map, (10, 7), init_state.boxes))
    list.append(State(init_state.level_map, (9, 7), init_state.boxes))
    list.append(State(init_state.level_map, (9, 6), init_state.boxes))
    list.append(State(init_state.level_map, (9, 5), init_state.boxes))
    list.append(State(init_state.level_map, (9, 4), init_state.boxes))

    return list
