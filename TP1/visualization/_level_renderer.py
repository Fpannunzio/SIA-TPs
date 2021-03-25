from typing import List

import pygame
from TP1 import visualization as viz_constants
from TP1.state import State
from TP1.map import Tile


class LevelRenderer:
    def __init__(self, level_map: List[List[Tile]]):
        self.structure: List[List[Tile]] = level_map
        self.width = len(level_map[0]) * viz_constants.SPRITESIZE
        self.height = len(level_map) * viz_constants.SPRITESIZE

    def render(self, window, textures, state: State) -> None:
        self._merge_map_with_state(state)
        self._render(window, textures)
        self._restore_map(state)

    def _merge_map_with_state(self, state: State):
        for (x, y) in state.boxes:
            tile_type: Tile = self.structure[y][x]

            if tile_type == Tile.AIR:
                tile_type = Tile.BOX

            elif tile_type == Tile.TARGET:
                tile_type = Tile.TARGET_FILLED

            else:
                raise RuntimeError(f'Box located in invalid place {(x, y)}')

            self.structure[y][x] = tile_type

    def _restore_map(self, state: State):
        for (x, y) in state.boxes:
            tile: Tile = self.structure[y][x]

            if tile == Tile.BOX:
                tile = Tile.AIR

            elif tile == Tile.TARGET_FILLED:
                tile = Tile.TARGET

            else:
                raise RuntimeError(f'Box located in invalid place {(x, y)}')

            self.structure[y][x] = tile

    def _render(self, window, textures):
        for y in range(len(self.structure)):
            for x in range(len(self.structure[y])):
                if self.structure[y][x] in textures:
                    window.blit(textures[self.structure[y][x]], (x * viz_constants.SPRITESIZE, y * viz_constants.SPRITESIZE))
                else:
                    if self.structure[y][x] == Tile.TARGET_FILLED:
                        pygame.draw.rect(window, (0, 255, 0), (x * viz_constants.SPRITESIZE, y * viz_constants.SPRITESIZE, viz_constants.SPRITESIZE, viz_constants.SPRITESIZE))
                    else:
                        pygame.draw.rect(window, viz_constants.WHITE, (x * viz_constants.SPRITESIZE, y * viz_constants.SPRITESIZE, viz_constants.SPRITESIZE, viz_constants.SPRITESIZE))
