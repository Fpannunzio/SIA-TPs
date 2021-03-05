from typing import Tuple, List

from tile_type import TileType


class State:

    def __init__(self, level_map: List[List[TileType]], player_pos: Tuple[int, int], boxes_pos: List[Tuple[int, int]]):
        self.level_map = level_map
        self.player: Tuple[int, int] = player_pos  # (x, y)
        self.boxes = boxes_pos  # [(x, y)]
