from typing import List, Set, Optional

from map import Tile, Position, Map
from state import State


def load_initial_state(level_name: str) -> State:
    level_map: List[List[Tile]] = []
    player_pos: Optional[Position] = None
    box_positions: Set[Position] = set()
    targets: Set[Position] = set()
    target_count: int = 0
    max_width: int = 0

    with open("assets/levels/" + level_name) as level_file:
        rows = level_file.read().split('\n')

        for y in range(len(rows)):
            level_row = []

            if len(rows[y]) > max_width:
                max_width = len(rows[y])

            for x in range(len(rows[y])):
                try:
                    tile = Tile(rows[y][x])
                except ValueError:
                    raise RuntimeError(f'Invalid character {rows[y][x]}. Only {list(Tile)} allowed')

                if tile == Tile.PLAYER:
                    if player_pos:
                        raise RuntimeError('Two players were found on map. Only one allowed')
                    else:
                        player_pos = Position(x, y)
                        tile = Tile.AIR

                elif tile == Tile.BOX:
                    box_positions.add(Position(x, y))
                    tile = Tile.AIR

                elif tile == Tile.TARGET:
                    targets.add(Position(x, y))
                    target_count += 1

                elif tile == Tile.TARGET_FILLED:
                    box_positions.add(Position(x, y))
                    target_count += 1
                    tile = Tile.TARGET

                level_row.append(tile)

            level_map.append(level_row)

        if not player_pos:
            raise RuntimeError(f'No player found on map. Please include a {Tile.PLAYER.value} character')

        if len(box_positions) == 0:
            raise RuntimeError(f'No boxes included. Please use the {Tile.BOX.value} character')

        if len(box_positions) != target_count:
            raise RuntimeError('Box target count and box count differ. Please make them equal')

        _normalize_level_map(level_map, max_width)

    return State(Map(level_map, targets), player_pos, frozenset(box_positions), target_count)


def _normalize_level_map(level_map: List[List[Tile]], max_width: int) -> None:
    if len(level_map[-1]) == 0:
        level_map.pop()  # Remove last line if empty

    for row in level_map:
        while len(row) < max_width:
            row.append(Tile.AIR)
