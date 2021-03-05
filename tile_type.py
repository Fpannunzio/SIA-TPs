from enum import Enum


class TileType(Enum):
    WALL = 'X'
    BOX = '*'
    TARGET = '.'
    TARGET_FILLED = '+'
    AIR = ' '
    PLAYER = '@'

    def is_filled(self) -> bool:
        return self in (TileType.WALL, TileType.BOX, TileType.TARGET_FILLED)

    def is_empty(self) -> bool:
        return self in (TileType.AIR, TileType.TARGET)
