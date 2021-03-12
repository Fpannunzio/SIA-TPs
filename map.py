from enum import Enum
from typing import List, NamedTuple

Position = NamedTuple('Position', [('x', int), ('y', int)])


class Tile(Enum):
    WALL = 'X'
    BOX = '*'
    TARGET = '.'
    TARGET_FILLED = '+'
    AIR = ' '
    PLAYER = '@'

    def is_empty(self) -> bool:
        return self in (Tile.AIR, Tile.TARGET)


class Map:

    def __init__(self, map: List[List[Tile]]) -> None:
        self.map = map

    def get_tile(self, pos: Position):

        # Out Of Bounds
        if pos.y < 0 or pos.x < 0 or pos.y >= len(self.map) or pos.x >= len(self.map[0]):
            return Tile.WALL

        return self.map[pos.y][pos.x]

    def is_empty(self, pos: Position) -> bool:
        return self.get_tile(pos) in (Tile.AIR, Tile.TARGET)
