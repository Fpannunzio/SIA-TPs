from enum import Enum
from typing import Set, Iterator, FrozenSet

from map import Tile, Position, Map


class MoveDir(Enum):

    UP = Position(0, -1)
    DOWN = Position(0, 1)
    LEFT = Position(-1, 0)
    RIGHT = Position(1, 0)

    def get_new_pos(self, pos: Position) -> Position:
        return Position(pos.x + self.value.x, pos.y + self.value.y)


class State:

    def __init__(self, level_map: Map, player_pos: Position, boxes: FrozenSet[Position], targets_remaining: int):
        self.level_map: Map = level_map
        self.player_pos: Position = player_pos  # (x, y)
        self.boxes: FrozenSet[Position] = boxes  # [(x, y)]
        self.targets_remaining: int = targets_remaining
        self.lost: bool = False

    def get_valid_player_moves(self) -> Iterator[MoveDir]:
        return filter(self.is_valid_player_move, MoveDir)

    def is_valid_player_move(self, move: MoveDir) -> bool:
        pos: Position = move.get_new_pos(self.player_pos)
        two_step_pos: Position = move.get_new_pos(pos)

        box_space: bool = pos in self.boxes
        empty_space: bool = self.level_map.is_empty(pos) and not box_space

        two_step_box_space: bool = two_step_pos in self.boxes
        two_step_empty_space: bool = self.level_map.is_empty(two_step_pos) and not two_step_box_space

        movable_box_space: bool = box_space and two_step_empty_space

        return empty_space or movable_box_space

    def move_player(self, move: MoveDir) -> 'State':

        if not self.is_valid_player_move(move):
            raise RuntimeError(f'Illegal player move {move}')

        new_state: State = self.copy()

        new_state.player_pos = move.get_new_pos(self.player_pos)

        if new_state.player_pos in self.boxes:
            new_state._move_box(move)

        return new_state

    # We assume valid input and state. That's why this function is private
    # State: player just moved to a box tile and the box could be moved in that direction.
    #
    # This method should only be called when a new State is being created. This is because self.boxed is
    # mutated after construction. If this is not honored, the class would cease to be immutable
    def _move_box(self, move: MoveDir) -> None:

        tmp_set: Set[Position] = set(self.boxes)

        box_new_position: Position = move.get_new_pos(self.player_pos)

        if self.get_player_tile() is Tile.TARGET:
            self.targets_remaining += 1

        tmp_set.remove(self.player_pos)

        if self.level_map.get_tile(box_new_position) is Tile.TARGET:
            self.targets_remaining -= 1

        tmp_set.add(box_new_position)

        self.boxes = frozenset(tmp_set)

        # If a box is in a wall corner that is not a target, the player cannot win from this state
        self.lost = self.is_wall_corner(box_new_position) and not self.level_map.get_tile(box_new_position) == Tile.TARGET

    def is_wall_corner(self, pos: Position) -> bool:

        up: bool = self.level_map.get_tile(MoveDir.UP.get_new_pos(pos)) == Tile.WALL
        down: bool = self.level_map.get_tile(MoveDir.DOWN.get_new_pos(pos)) == Tile.WALL
        left: bool = self.level_map.get_tile(MoveDir.LEFT.get_new_pos(pos)) == Tile.WALL
        right: bool = self.level_map.get_tile(MoveDir.RIGHT.get_new_pos(pos)) == Tile.WALL

        up_right = up and right
        up_left = up and left
        down_right = down and right
        down_left = down and left

        return up_right or up_left or down_right or down_left

    def get_player_tile(self) -> Tile:
        return self.level_map.get_tile(self.player_pos)

    def has_won(self) -> bool:
        return self.targets_remaining == 0

    def has_lost(self) -> bool:
        return self.lost

    def __eq__(self, value) -> bool:
        return isinstance(value, State) and\
               self.level_map == value.level_map and\
               self.player_pos == value.player_pos and\
               self.boxes == value.boxes

    def __hash__(self) -> int:
        return hash((self.player_pos, self.boxes))

    # Level stays the same across states
    def copy(self):
        return State(self.level_map, self.player_pos, self.boxes.copy(), self.targets_remaining)
