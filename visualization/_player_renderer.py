from typing import Tuple

import pygame
import visualization._viz_constants as viz_constants

from map import Tile, Position


class PlayerRenderer:

    def __init__(self, init_pos: Position):
        self.pos: Position = init_pos
        self.direction = viz_constants.DOWN

    def update_pos(self, new_pos: Position) -> None:
        if self.pos == new_pos:
            return

        if new_pos.x == self.pos.x + 1:
            self.direction = viz_constants.RIGHT

        elif new_pos.x == self.pos.x - 1:
            self.direction = viz_constants.LEFT

        elif new_pos.y == self.pos.y + 1:
            self.direction = viz_constants.DOWN

        elif new_pos.y == self.pos.y - 1:
            self.direction = viz_constants.UP

        else:
            raise RuntimeError(f'Invalid player move from {self.pos} to {new_pos}')

        self.pos = new_pos

    def render(self, window, textures) -> None:
        if self.direction == viz_constants.DOWN:
            top = 0
        elif self.direction == viz_constants.LEFT:
            top = viz_constants.SPRITESIZE
        elif self.direction == viz_constants.RIGHT:
            top = viz_constants.SPRITESIZE * 2
        elif self.direction == viz_constants.UP:
            top = viz_constants.SPRITESIZE * 3
        else:
            raise RuntimeError(f'Invalid direction value ({self.direction})')

        area_player = pygame.Rect((0, top), (32, 32))
        window.blit(textures[Tile.PLAYER], (self.pos.x * viz_constants.SPRITESIZE, self.pos.y * viz_constants.SPRITESIZE), area=area_player)
