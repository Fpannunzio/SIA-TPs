from typing import Tuple, List

from tile_type import TileType


class State:

    def __init__(self, level_map: List[List[TileType]], player_pos: Tuple[int, int], boxes_pos: List[Tuple[int, int]], target_remaining: int):
        self.level_map = level_map
        self.player: Tuple[int, int] = player_pos  # (x, y)
        self.boxes = boxes_pos  # [(x, y)]
        self.target_remaining = target_remaining

    def has_won(self) -> bool:
        return self.target_remaining == 0

    def __eq__(self, value):
        return self.player == value.player and set(self.boxes) == set(value.boxes)

    def copy(self):
        return State(self.level_map, self.player, self.boxes.copy(), self.target_remaining)