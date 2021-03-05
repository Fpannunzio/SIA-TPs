from typing import List

import pygame
import visualization._viz_constants as viz_constants
from state import State
from tile_type import TileType


class Level:
    def __init__(self, level_map: List[List[TileType]]):
        self.structure = level_map
        self.width = len(level_map[0]) * viz_constants.SPRITESIZE
        self.height = len(level_map) * viz_constants.SPRITESIZE

    def render(self, window, textures, state: State) -> None:
        self._merge_map_with_state(state)
        self._render(window, textures)
        self._restore_map(state)

    def _merge_map_with_state(self, state: State):
        for (x, y) in state.boxes:
            tile_type: TileType = TileType(self.structure[y][x])

            if tile_type == TileType.AIR:
                tile_type = TileType.BOX

            elif tile_type == TileType.TARGET:
                tile_type = TileType.TARGET_FILLED

            else:
                raise RuntimeError(f'Box located in invalid place {(x, y)}')

            self.structure[y][x] = tile_type

    def _restore_map(self, state: State):
        for (x, y) in state.boxes:
            tile_type: TileType = TileType(self.structure[y][x])

            if tile_type == TileType.BOX:
                tile_type = TileType.AIR

            elif tile_type == TileType.TARGET_FILLED:
                tile_type = TileType.TARGET

            else:
                raise RuntimeError(f'Box located in invalid place {(x, y)}')

            self.structure[y][x] = tile_type

    def _render(self, window, textures):
        for y in range(len(self.structure)):
            for x in range(len(self.structure[y])):
                if self.structure[y][x] in textures:
                    window.blit(textures[self.structure[y][x]], (x * viz_constants.SPRITESIZE, y * viz_constants.SPRITESIZE))
                else:
                    if self.structure[y][x] == TileType.TARGET_FILLED:
                        pygame.draw.rect(window, (0, 255, 0), (x * viz_constants.SPRITESIZE, y * viz_constants.SPRITESIZE, viz_constants.SPRITESIZE, viz_constants.SPRITESIZE))
                    else:
                        pygame.draw.rect(window, viz_constants.WHITE, (x * viz_constants.SPRITESIZE, y * viz_constants.SPRITESIZE, viz_constants.SPRITESIZE, viz_constants.SPRITESIZE))
