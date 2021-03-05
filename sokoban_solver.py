from typing import List, Tuple, Union

from dfs import dfs
from visualization.game_renderer import GameRenderer
import sys

from state import State
from tile_type import TileType


def main(level_name: str, strategy: str):

    # Load initial state from level file selected
    initial_state: State = load_initial_state(level_name)

    # Solve Sokoban using selected strategy
    states: List[State] = solve_sokoban(strategy, initial_state)

    # Render Solution
    GameRenderer(states).render()


def load_initial_state(level_name: str) -> State:
    level_map: List[List[TileType]] = []
    player_pos: Union[Tuple[int, int], None] = None
    box_pos_list: List[Tuple[int, int]] = []
    target_count: int = 0
    max_width = 0

    with open("assets/levels/" + level_name) as level_file:
        rows = level_file.read().split('\n')

        for y in range(len(rows)):
            level_row = []
            if len(rows[y]) > max_width:
                max_width = len(rows[y])

            for x in range(len(rows[y])):
                # TODO: Catch exception when invalid value
                tile_type = TileType(rows[y][x])

                if tile_type == TileType.PLAYER:
                    if player_pos:
                        raise RuntimeError('Two players were found on map. Only one allowed')
                    else:
                        player_pos = (x, y)
                        tile_type = TileType.AIR

                elif tile_type == TileType.BOX:
                    box_pos_list.append((x, y))
                    tile_type = TileType.AIR

                elif tile_type == TileType.TARGET:
                    target_count += 1

                elif tile_type == TileType.TARGET_FILLED:
                    box_pos_list.append((x, y))
                    target_count += 1
                    tile_type = TileType.TARGET

                level_row.append(tile_type)

            level_map.append(level_row)

        if not player_pos:
            raise RuntimeError(f'No player found on map. Please include a {TileType.PLAYER.value} character')

        if len(box_pos_list) == 0:
            raise RuntimeError(f'No boxes included. Please use the {TileType.BOX.value} character')

        if len(box_pos_list) != target_count:
            raise RuntimeError('Box target count and box count differ. Please make them equal')

        _normalize_level_map(level_map, max_width)

    return State(level_map, player_pos, box_pos_list)


def _normalize_level_map(level_map: List[List[TileType]], max_width: int) -> None:
    if len(level_map[-1]) == 0:
        level_map.pop()  # Remove last line if empty

    for row in level_map:
        while len(row) < max_width:
            row.append(TileType.AIR)


# TODO: Replace with dictionary
def solve_sokoban(strategy: str, init_state: State) -> List[State]:
    if strategy == 'DFS':
        return dfs(init_state)

    elif strategy == 'BFS':
        pass  # TODO: bfs(init_state)

    elif strategy == 'IDDFS':
        pass  # TODO: iddfs(init_state)

    else:
        raise RuntimeError(f'Invalid Strategy {strategy}. Currently supported: [BFS, DFS, IDDFS]')


# Usage: python3 Sokoban [level_name] [solve_strategy]
if __name__ == "__main__":
    argv = sys.argv

    level_name_arg: str = (argv[1] if len(argv) >= 2 else "level.txt")
    strategy_arg: str = (argv[2] if len(argv) >= 3 else "DFS")

    main(level_name_arg, strategy_arg)
