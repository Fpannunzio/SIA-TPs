import math
from abc import ABC, abstractmethod
from typing import Callable, Optional, Tuple

import numpy as np

ActivationFunction = Callable[[float], float]


class Perceptron(ABC):

    DEFAULT_MAX_ITERATION       : int = 10000
    DEFAULT_SOFT_RESET_THRESHOLD: int = 1000

    @staticmethod
    def calculate_weighted_sum(point: np.ndarray, w: np.array) -> float:
        return np.sum(point * w)

    @staticmethod
    def with_identity_dimension(points: np.ndarray) -> np.ndarray:
        return np.insert(points, 0, 1, axis=1)

    def __init__(self, l_rate: float, input_count: int, activation: ActivationFunction,
                 max_training_iteration: Optional[int] = None, soft_reset_threshold: Optional[int] = None, ) -> None:

        self.l_rate: float = l_rate
        self.activation: ActivationFunction = activation
        self.w: np.ndarray = np.random.uniform(-1, 1, input_count + 1)  # array de n + 1 puntos con dist. Uniforme([-1, 1))
        self.error: Optional[float] = None

        # Training
        self.training_w: np.ndarray = np.copy(self.w)
        self.training_iteration: int = 0
        self.iters_since_soft_reset: int = 0
        self.max_training_iteration = (max_training_iteration if max_training_iteration is not None else Perceptron.DEFAULT_MAX_ITERATION)
        self.soft_reset_threshold = (soft_reset_threshold if soft_reset_threshold is not None else Perceptron.DEFAULT_SOFT_RESET_THRESHOLD)

    @abstractmethod
    def calculate_delta_weight(self, point: np.ndarray, point_value: float, weighted_sum: float) -> np.ndarray:
        pass

    @abstractmethod
    def calculate_error(self, training_points: np.ndarray, training_values: np.ndarray, w: np.ndarray) -> float:
        pass

    def soft_training_reset(self) -> None:
        self.training_w = np.random.uniform(-1, 1, len(self.training_w))
        self.iters_since_soft_reset = 0

    def hard_training_reset(self) -> None:
        self.soft_training_reset()
        self.training_iteration = 0

    def has_training_ended(self) -> bool:
        return self.error is not None and (self.error <= 0 or self.training_iteration >= self.max_training_iteration)

    # No cambiar los training points en el medio del entrenamiento
    # Antes de empezar un nuevo entrenamiento hacer un hard_training_reset
    # Asume que los training_points ya tienen la columna identidad
    def do_training_iteration(self, training_points: np.ndarray, training_values: np.ndarray) -> None:
        if self.iters_since_soft_reset > self.soft_reset_threshold * len(training_points):
            self.soft_training_reset()

        # Seleccionamos un punto del training set al azar
        point: int = np.random.randint(0, len(training_points))  # random int intervalo [0, n + 1) => [0, n]

        # Actualizamos el valor del vector peso
        self.training_w += self.calculate_delta_weight(
            training_points[point],
            training_values[point],
            Perceptron.calculate_weighted_sum(training_points[point], self.training_w)
        )

        # Actualizamos el error
        current_error: float = self.calculate_error(training_points, training_values, self.training_w)
        if self.error is None or current_error < self.error:
            self.error = current_error
            self.w = np.copy(self.training_w)

        self.training_iteration += 1

    def train(self, training_points: np.ndarray, training_values: np.ndarray,
              status_callback: Optional[Callable[[np.ndarray], None]] = None, insert_identity_column: bool = True) -> Tuple[int, np.ndarray]:

        if insert_identity_column:
            training_points = Perceptron.with_identity_dimension(training_points)

        while not self.has_training_ended():
            self.do_training_iteration(training_points, training_values)
            if status_callback:
                status_callback(self.training_w)

        # Retorno al estado inicial y devuelvo el training_w final y la cantidad de training_iterations
        ret: Tuple[int, np.ndarray] = (self.training_iteration, self.training_w)
        self.hard_training_reset()

        return ret

    # Asume que el punto tiene la columna identidad
    def _predict(self, point: np.ndarray, w: np.ndarray) -> float:
        return self.activation(Perceptron.calculate_weighted_sum(point, w))

    def predict(self, point: np.ndarray, insert_identity_column: bool = False) -> float:
        if insert_identity_column:
            point = np.insert(point, 0, 1)
        return self.activation(Perceptron.calculate_weighted_sum(point, self.w))

    # TODO(tobi): Alguno sabe una mejor manera? - Es mejor con fromiter o array
    def predict_points(self, points: np.ndarray, insert_identity_column: bool = True) -> np.ndarray:
        if insert_identity_column:
            points = Perceptron.with_identity_dimension(points)
        return np.array(map(self.predict, points))

    # Retorna los puntos que fueron predecidos incorrectamente
    # TODO(tobi): Alguno sabe una mejor manera? - Es mejor con fromiter o array
    def validate_points(self, points: np.ndarray, values: np.ndarray, insert_identity_column: bool = True) -> np.ndarray:
        if insert_identity_column:
            points = Perceptron.with_identity_dimension(points)

        return np.array([points[point] for point in range(len(points)) if not math.isclose(self.predict(points[point]), values[point])])

    def is_validation_successful(self, points: np.ndarray, values: np.ndarray, insert_identity_column: bool = True) -> bool:
        if insert_identity_column:
            points = Perceptron.with_identity_dimension(points)

        for i in range(len(points)):
            if not math.isclose(self.predict(points[i]), values[i]):
                return False
        return True


class SimplePerceptron(Perceptron):

    def __init__(self, l_rate: float, input_count: int,
                 max_training_iteration: Optional[int] = None, soft_reset_threshold: Optional[int] = None) -> None:
        # Activation Function = funcion signo
        super().__init__(l_rate, input_count, np.sign, max_training_iteration, soft_reset_threshold)

    def calculate_delta_weight(self, point: np.ndarray, point_value: float, weighted_sum: float) -> np.ndarray:
        return self.l_rate * (point_value - self.activation(weighted_sum)) * point

    # TODO(tobi): wat
    def calculate_error(self, training_points: np.ndarray, training_values: np.ndarray, w: np.ndarray) -> float:
        if sum(abs(training_values[point] - self._predict(training_points[point], w)) for point in range(len(training_points))) == 0:
            print('bien')

        return sum(abs(training_values[point] - self._predict(training_points[point], w)) for point in range(len(training_points)))


class LinearPerceptron(Perceptron):

    def __init__(self, l_rate: float, input_count: int, max_training_iteration: Optional[int] = None,
                 soft_reset_threshold: Optional[int] = None) -> None:
        # Activation Function = funcion identidad
        super().__init__(l_rate, input_count, lambda x: x, max_training_iteration, soft_reset_threshold)

    def calculate_delta_weight(self, point: np.ndarray, point_value: float, weighted_sum: float) -> np.ndarray:
        return self.l_rate * (point_value - self.activation(weighted_sum)) * point

    def calculate_error(self, training_points: np.ndarray, training_values: np.ndarray, w: np.ndarray) -> float:
        return sum([0.5 * (abs(training_values[point] - self._predict(training_points[point], w)) ** 2)
                    for point in range(len(training_points))])


class NonLinearPerceptron(Perceptron):

    def __init__(self, l_rate: float, input_count: int, activation: ActivationFunction, activation_derivative: ActivationFunction,
                 max_training_iteration: Optional[int] = None, soft_reset_threshold: Optional[int] = None) -> None:
        super().__init__(l_rate, input_count, activation, max_training_iteration, soft_reset_threshold)
        self.activation_derivative = activation_derivative

    def calculate_delta_weight(self, point: np.ndarray, point_value: float, weighted_sum: float) -> np.ndarray:
        return self.l_rate * (point_value - self.activation(weighted_sum)) * point * self.activation_derivative(weighted_sum)

    def calculate_error(self, training_points: np.ndarray, training_values: np.ndarray, w: np.ndarray) -> float:
        return sum([0.5 * (abs(training_values[point] - self._predict(training_points[point], w)) ** 2)
                    for point in range(len(training_points))])
