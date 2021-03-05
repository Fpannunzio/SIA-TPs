import time
from typing import List

import pygame
import visualization._viz_constants as viz_constants
from visualization._level import Level
from visualization._player import Player
from state import State
from tile_type import TileType


class GameRenderer:

    @staticmethod
    def _load_textures():
        return {
            TileType.WALL: pygame.image.load('assets/images/wall.png').convert_alpha(),
            TileType.BOX: pygame.image.load('assets/images/box.png').convert_alpha(),
            TileType.TARGET: pygame.image.load('assets/images/target.png').convert_alpha(),
            TileType.TARGET_FILLED: pygame.image.load('assets/images/valid_box.png').convert_alpha(),
            TileType.PLAYER: pygame.image.load('assets/images/player_sprites.png').convert_alpha()
        }

    def __init__(self, states: List[State]):

        if len(states) == 0:
            raise RuntimeError('State list must have at least one state to correctly render Sokoban')

        pygame.init()
        pygame.display.set_caption("Sokoban Game")

        self.window = pygame.display.set_mode((viz_constants.WINDOW_WIDTH, viz_constants.WINDOW_HEIGHT))
        self.textures = self._load_textures()

        self.states = states
        init_state = self.states[0]

        self.player = Player(init_state.player)
        self.level = Level(init_state.level_map)

        self.board = pygame.Surface((self.level.width, self.level.height))

    def render(self):
        for state in self.states:
            self.update_screen(state)
            print(state.target_remaining, state.boxes)
            time.sleep(1)

    def update_screen(self, state: State):
        pygame.draw.rect(self.board, viz_constants.WHITE, (0, 0, self.level.width * viz_constants.SPRITESIZE, self.level.height * viz_constants.SPRITESIZE))
        pygame.draw.rect(self.window, viz_constants.WHITE, (0, 0, viz_constants.WINDOW_WIDTH, viz_constants.WINDOW_HEIGHT))

        self.level.render(self.board, self.textures, state)
        self.player.update_pos(state.player)
        self.player.render(self.board, self.textures)

        pox_x_board = (viz_constants.WINDOW_WIDTH / 2) - (self.board.get_width() / 2)
        pos_y_board = (viz_constants.WINDOW_HEIGHT / 2) - (self.board.get_height() / 2)
        self.window.blit(self.board, (pox_x_board, pos_y_board))

        pygame.display.flip()
