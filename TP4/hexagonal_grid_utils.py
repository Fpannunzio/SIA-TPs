from typing import NamedTuple

import numpy as np

OffsetCoord = NamedTuple('OffsetCoord', [('row', int), ('col', int)])
CubeCord = NamedTuple('CubeCord', [('x', int), ('y', int), ('z', int)])


# https://www.redblobgames.com/grids/hexagons/
def cube_to_oddr(hex: CubeCord) -> OffsetCoord:
    col: int = hex.x + hex.z // 2
    row: int = hex.z
    return OffsetCoord(col, row)


def offset_to_cube(hex: OffsetCoord) -> CubeCord:
    x: int = hex.col - hex.row // 2
    z: int = hex.row
    y: int = -x - z
    return CubeCord(x, y, z)


def offset_distance(hex_a: OffsetCoord, hex_b: OffsetCoord):
    ac: CubeCord = offset_to_cube(hex_a)
    bc: CubeCord = offset_to_cube(hex_b)
    return cube_distance(ac, bc)


def cube_distance(hex_a: CubeCord, hex_b: CubeCord):
    return (abs(hex_a.x - hex_b.x) + abs(hex_a.y - hex_b.y) + abs(hex_a.z - hex_b.z)) / 2


def generate_indexes_matrix(g_size: int) -> np.ndarray:
    grid = np.zeros((g_size, g_size, 2), dtype=np.int32)
    grid[:, :, 1] = np.arange(g_size)
    grid = grid.reshape((g_size ** 2, 2))
    grid[:, 0] = np.repeat(np.arange(g_size), g_size)
    return grid
