import math
from abc import ABC, abstractmethod
from typing import Callable, List, NamedTuple, Optional, TypeVar

import attr
import numpy as np

from hexagonal_grid_utils import generate_indexes_matrix, offset_distance, OffsetCoord


@attr.s(auto_attribs=True)
class Index:
    x: int
    y: int

    def get_neuron_number(self, k: int) -> int:
        return self.x * k + self.y

# Generic Internal Variable
_T = TypeVar('_T')


def _assert_not_none(obj: Optional[_T]) -> _T:
    if obj is None:
        raise TypeError()
    return obj


class _Neuron:
    def __init__(self, distance: Callable[[np.ndarray, np.ndarray], float], input_count: int,
                 init_w: np.ndarray = None):
        self.distance = distance
        self.input_count = input_count
        self.training_winning_count = 0
        self.w: np.ndarray = np.random.uniform(-1, 1, self.input_count) if init_w is None else init_w

    def __repr__(self) -> str:
        return f'{self.w.__str__()}\n'

    def calculate_distance(self, point: np.ndarray) -> float:
        return self.distance(self.w, point)

    def update_weights(self, learning_rate: float, selected_point: np.ndarray) -> None:
        self.w += learning_rate * (selected_point - self.w)


@attr.s(auto_attribs=True)
class GridBaseConfiguration:
    input_count: Optional[int] = None
    learning_rate: Optional[float] = None
    radius: Optional[float] = None
    k: Optional[int] = None
    distance: Callable[[np.ndarray, np.ndarray], float] = None

#     TODO(tobi): Validar parametros


class KohonenGrid(ABC):
    def __init__(self, grid_config: GridBaseConfiguration, initial_weights: np.ndarray = None):
        self.radius = _assert_not_none(grid_config.radius)
        self.learning_rate = 1
        self.k = _assert_not_none(grid_config.k)
        self.grid: List[List[_Neuron]] = self._generate_grid(self.k, grid_config.distance, grid_config.input_count, initial_weights)

    def __str__(self) -> str:
        return self.grid.__str__()

    @staticmethod
    def _generate_grid(k: int, distance: Callable[[np.ndarray, np.ndarray], float],
                       input_count: int, initial_weights: np.ndarray = None) -> List[List[_Neuron]]:
        aux: List[List[_Neuron]] = []
        for i in range(k):
            aux.append([])
            for j in range(k):
                aux[i].append(_Neuron(distance, input_count, initial_weights[
                    np.random.randint(len(initial_weights))] if initial_weights is not None else None))

        return aux

    @abstractmethod
    def _get_near_neurons_indexes(self, index: Index, radius: float = 1.0) -> List[Index]:
        pass

    def update_near_neurons_weights(self, near_neurons_indexes: List[Index], current_point: np.ndarray) -> None:
        for i in range(len(near_neurons_indexes)):
            self.grid[near_neurons_indexes[i].x][near_neurons_indexes[i].y].update_weights(self.learning_rate, current_point)

    def train(self, epochs: int, initial_values: np.ndarray) -> None:
        index: Index
        near_neurons_indexes: List[Index]

        for epoch in range(epochs):
            for current_value in range(np.size(initial_values, axis=0)):

                index = self.predict_value(initial_values[current_value])

                self.update_near_neurons_weights(self._get_near_neurons_indexes(index, self.radius), initial_values[current_value])
                self.grid[index.x][index.y].training_winning_count += 1

            self.learning_rate = 1 / (epoch + 1)

    #     TODO(tobi): Donde se actualiza el radio y el learning rate??
    #         radio PUEDE ser constante, pero l_rate debe tender a 0

    def _get_near_neurons_mean_distance(self, index: Index, near_neurons_indexes: List[Index]) -> float:
        # lista de distancias entre los pesos de la neurona 'index' y sus vecinos
        return np.array(
            [
                self.grid[index.x][index.y].calculate_distance(
                    self.grid[near_neurons_indexes[i].x][near_neurons_indexes[i].y].w
                )
                for i in range(len(near_neurons_indexes))
            ]
        ).mean()

    # Matriz U
    def get_near_neurons_mean_distances_matrix(self) -> np.ndarray:
        mean_distance_matrix: np.ndarray = np.zeros((self.k, self.k))

        for i in range(self.k):
            for j in range(self.k):
                # Conseguir la media de las distancias de todos los nodos vecinos, no se le pasa el
                # radio a _get_near_neurons_indexes porque
                # quiero que sean sus vecinos próximos, de radio = 1
                mean_distance_matrix[i][j] = self._get_near_neurons_mean_distance(
                    Index(i, j),
                    self._get_near_neurons_indexes(Index(i, j))
                )

        return mean_distance_matrix

    def predict_value(self, value: np.ndarray):
        index = Index(0, 0)

        min_distance: float = math.inf
        current_distance: float = math.inf

        for i in range(self.k):
            for j in range(self.k):
                current_distance = self.grid[i][j].calculate_distance(value)

                if current_distance < min_distance:
                    min_distance = current_distance
                    index = Index(i, j)
        return index

    def predict(self, values: np.ndarray) -> np.ndarray:
        ans: List[int] = []
        for value in values:
            ans.append(self.predict_value(value).get_neuron_number(self.k))

        return np.array(ans)


class KohonenQuadraticGrid(KohonenGrid):

    def __init__(self, grid_config: GridBaseConfiguration, initial_weights: np.ndarray = None):
        super().__init__(grid_config, initial_weights)

    def _get_near_neurons_indexes(self, index: Index, radius: float = 1.0) -> List[Index]:
        grid = np.zeros((self.k, self.k, 2), dtype=np.int32)
        grid[:, :, 1] = np.arange(self.k)
        grid = grid.reshape((self.k ** 2, 2))
        grid[:, 0] = np.repeat(np.arange(self.k), self.k)
        mask = np.linalg.norm(grid[:] - np.array([index.x, index.y]), axis=1) <= radius
        return [Index(value[0], value[1]) for value in grid[mask]]


class KohonenHexagonalGrid(KohonenGrid):

    def __init__(self, grid_config: GridBaseConfiguration, initial_weights: np.ndarray = None):
        super().__init__(grid_config, initial_weights)

    def _get_near_neurons_indexes(self, index: Index, radius: float = 1.0) -> List[Index]:
        grid = generate_indexes_matrix(self.k)
        mask = np.apply_along_axis(lambda arr, point: offset_distance(OffsetCoord(arr[0], arr[1]), point), 1, grid,
                                   OffsetCoord(index.x, index.y))
        mask = mask <= radius
        return [Index(value[0], value[1]) for value in grid[mask]]
