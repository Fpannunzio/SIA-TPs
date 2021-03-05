from typing import Tuple, List

from tile_type import TileType


class State:

    def __init__(self, level_map: List[List[TileType]], player_pos: Tuple[int, int], boxes: List[Tuple[int, int]], targets_remaining: int):
        self.level_map = level_map
        self.player_pos: Tuple[int, int] = player_pos  # (x, y)
        self.boxes = boxes  # [(x, y)]
        self.targets_remaining = targets_remaining

    def has_won(self) -> bool:
        return self.targets_remaining == 0

    def __eq__(self, value):
        return self.player_pos == value.player_pos and set(self.boxes) == set(value.boxes)

    # Level stays the same across states
    def copy(self):
        return State(self.level_map, self.player_pos, self.boxes.copy(), self.targets_remaining)