from typing import List, Tuple

from state import State
from tile_type import TileType

def process_move(state: State, movement_direction: Tuple[int, int]) -> State:

    if is_valid_direction(movement_direction):
        new_position: Tuple[int, int] = (
            state.player_pos[0] + movement_direction[0], 
            state.player_pos[1] + movement_direction[1]
            )
        
        new_state = state.copy()

        if get_tile(state.level_map, new_position).is_empty() and not new_position in state.boxes:
            new_state.player_pos = new_position
            return new_state
            
        if new_position in state.boxes:
            if move_box(new_state, new_position, movement_direction):
                new_state.player_pos = new_position
                return new_state
    return None


def is_valid_direction(direction: Tuple[int, int]) -> bool:
    return (abs(direction[0]) == 1 and direction[1] == 0) or (abs(direction[1]) == 1 and direction[0] == 0)

def move_box(new_state: State, player_new_pos: Tuple[int, int], movement_direction: Tuple[int, int]) -> List[Tuple[int, int]]:
    
    box_new_position: Tuple[int, int] = (
        player_new_pos[0] + movement_direction[0], 
        player_new_pos[1] + movement_direction[1]
        )
    
    if get_tile(new_state.level_map, box_new_position).is_empty() and not box_new_position in new_state.boxes:
        
        if get_tile(new_state.level_map, player_new_pos) is TileType.TARGET:
            new_state.targets_remaining += 1
            
        new_state.boxes.remove(player_new_pos)
        
        if get_tile(new_state.level_map, box_new_position) is TileType.TARGET:
            new_state.targets_remaining -= 1

        new_state.boxes.append(box_new_position)
        
        return True
    return False
        

def get_tile(level_map:List[List[TileType]], pos: Tuple[int, int]):
    return level_map[pos[1]][pos[0]]