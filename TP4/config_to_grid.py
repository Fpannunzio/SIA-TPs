from typing import Dict, Callable

import numpy as np
import pandas
from schema import Schema, And, Or
from sklearn.preprocessing import StandardScaler

from TP4.config import Param, Config
from TP4.grid import Grid, QuadraticGrid, HexagonalGrid, GridBaseConfiguration

GridFactory = Callable[[], Grid]
_GridFactoryBuilder = Callable[[GridBaseConfiguration], Callable[[], Grid]]


def get_normalized_values(file_name: str) -> np.ndarray:
    file = pandas.read_csv(file_name)

    return StandardScaler().fit_transform(file.values[:, 1:])


def get_grid(base_grid_params: Param, input_count: int) -> Grid:
    grid_factory: GridFactory = get_grid_factory(base_grid_params, input_count)
    return grid_factory()


def _validate_base_grid_params(grid_params: Param) -> Param:
    return Config.validate_param(grid_params, Schema({
        'connection': And(str, Or(*tuple(grid_factory_builder_map.keys()))),
        'distance_function': And(str, Or(*tuple(_distance_function_map.keys()))),
        'radius': And(int, lambda i: i > 0),
        'learning_rate': And(float, lambda i: i > 0),
        'k': And(int, lambda i: i > 0)
    }, ignore_extra_keys=True))


def _build_base_grid_config(grid_params: Param, input_count: int) -> GridBaseConfiguration:
    ret: GridBaseConfiguration = GridBaseConfiguration()
    if input_count is not None: ret.input_count = input_count
    if grid_params['radius'] is not None: ret.radius = grid_params['radius']
    if grid_params['learning_rate'] is not None: ret.learning_rate = grid_params['learning_rate']
    if grid_params['k'] is not None: ret.k = grid_params['k']
    if grid_params['distance_function'] is not None: ret.distance = _distance_function_map[
        grid_params['distance_function']]

    return ret


def get_grid_factory(base_grid_params: Param, input_count: int) -> GridFactory:
    base_grid_params = _validate_base_grid_params(base_grid_params)

    base_grid_config: GridBaseConfiguration = _build_base_grid_config(base_grid_params, input_count)

    factory_builder: _GridFactoryBuilder = grid_factory_builder_map[base_grid_params['connection']]

    return factory_builder(base_grid_config)


def _get_quadratic_grid(base_config: GridBaseConfiguration) -> Callable[[], Grid]:
    return lambda: QuadraticGrid(base_config)


def _get_hexagonal_grid(base_config: GridBaseConfiguration) -> Callable[[], Grid]:
    return lambda: HexagonalGrid(base_config)


def _euclidean_distance(point1: np.ndarray, point2: np.ndarray) -> float:
    return np.linalg.norm(point1 - point2)


grid_factory_builder_map: Dict[str, _GridFactoryBuilder] = {
    'quadratic': _get_quadratic_grid,
    'hexagonal': _get_hexagonal_grid
}

_distance_function_map: Dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    'euclidean': _euclidean_distance
}
