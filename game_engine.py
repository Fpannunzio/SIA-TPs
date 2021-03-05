from typing import List, Tuple

from state import State
from tile_type import TileType

def process_move(init_state: State, movement_direction: Tuple[int, int]) -> State:

    if is_valid_direction(movement_direction):
        new_position: Tuple[int, int] = (
            init_state.player[0] + movement_direction[0], 
            init_state.player[1] + movement_direction[1]
            )
        
        new_state = init_state.copy()

        if get_tile(init_state.level_map, new_position[0], new_position[1]).is_empty() and not is_position_box(init_state.boxes, new_position):
            new_state.player = new_position
            return new_state
        if is_position_box(init_state.boxes, new_position):
            if move_box(new_state, new_position, movement_direction):
                new_state.player = new_position
                return new_state

    return None


def is_valid_direction(direction: Tuple[int, int]) -> bool:
    return (abs(direction[0]) == 1 and direction[1] == 0) or (abs(direction[1]) == 1 and direction[0] == 0)

def is_position_box(boxes_pos: List[Tuple[int, int]], new_pos: Tuple[int, int]):
    
    return new_pos in boxes_pos
    # for box_pos in boxes_pos:
    #     if(new_pos[0] == box_pos[0] and new_pos[1] == box_pos[1]):
    #         print('Trueeee')
    #         return True
    # return False

def move_box(current_state: State, player_new_pos: Tuple[int, int], direction: Tuple[int, int]) -> List[Tuple[int, int]]:
    
    box_new_position: Tuple[int, int] = (
        player_new_pos[0] + direction[0], 
        player_new_pos[1] + direction[1]
        )
    
    if get_tile(current_state.level_map, box_new_position[0], box_new_position[1]).is_empty() and not is_position_box(current_state.boxes, box_new_position):
        

        if get_tile(current_state.level_map, player_new_pos[0], player_new_pos[1]) is TileType.TARGET:
            current_state.target_remaining += 1
            
        current_state.boxes.remove(player_new_pos)
        
        if get_tile(current_state.level_map, box_new_position[0], box_new_position[1]) is TileType.TARGET:
            current_state.target_remaining -= 1

        current_state.boxes.append(box_new_position)
        
        return True
    return False
        

def get_tile( level_map:List[List[TileType]], pos_x: int, pos_y: int):
    return level_map[pos_y][pos_x]