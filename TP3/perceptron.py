import math
from abc import ABC, abstractmethod
from typing import Callable, Optional

import numpy as np

ActivationFunction = Callable[[float], float]


class Perceptron(ABC):

    # TODO(tobi): Capaz cambiarlos a default value del constructor
    DEFAULT_MAX_ITERATION       : int = 10000
    DEFAULT_SOFT_RESET_THRESHOLD: int = 1000

    @staticmethod
    def calculate_weighted_sum(point: np.ndarray, w: np.array) -> float:
        return np.sum(point * w)

    @staticmethod
    def with_identity_dimension(points: np.ndarray) -> np.ndarray:
        return np.insert(points, 0, 1, axis=1)

    def soft_reset(self) -> None:
        self.w = np.random.uniform(-1, 1, len(self.training_points[0]))
        self.iters_since_soft_reset = 0

    # TODO(tobi): Puedo renombrar y por z?
    # len(x) = p = cantidad de puntos en el training set
    # len(x[0]) = n + 1 = dimension de los puntos + 1 (por la columna de 1 agregados, que representa la dimension de los resultados)
    def __init__(self, l_rate: float, activation: ActivationFunction, training_points: np.ndarray, training_values: np.ndarray,
                 max_iteration: Optional[int] = None, soft_reset_threshold: Optional[int] = None,
                 insert_identity_column: bool = True) -> None:

        self.l_rate: float = l_rate
        self.activation: ActivationFunction = activation
        self.training_points: np.ndarray = (Perceptron.with_identity_dimension(training_points) if insert_identity_column else training_points)
        self.training_values: np.ndarray = training_values
        self.max_iteration = (max_iteration if max_iteration is not None else Perceptron.DEFAULT_MAX_ITERATION)
        self.soft_reset_threshold = (soft_reset_threshold if soft_reset_threshold is not None else Perceptron.DEFAULT_SOFT_RESET_THRESHOLD)

        self.w: np.ndarray = np.random.uniform(-1, 1, len(self.training_points[0]))  # array de n + 1 puntos con dist. Uniforme([-1, 1))
        self.w_min: np.ndarray = np.copy(self.w)
        self.error: float = 1.0
        self.error_min: float = len(self.training_points) * 2
        self.iteration: int = 0
        self.iters_since_soft_reset: int = 0

    @abstractmethod
    def calculate_delta_weight(self, point: np.ndarray, point_value: float, weighted_sum: float) -> np.ndarray:
        pass

    @abstractmethod
    def calculate_error(self) -> float:
        return sum([0.5 * (abs(self.training_values[point] - self._predict(self.training_points[point])) ** 2) for point in range(len(self.training_points))])  # Good Default

    def has_training_ended(self) -> bool:
        return self.error <= 0 or self.iteration >= self.max_iteration

    def do_training_iteration(self) -> None:
        if self.iters_since_soft_reset > self.soft_reset_threshold * len(self.training_points):
            self.soft_reset()

        # Seleccionamos un punto del training set al azar
        point: int = np.random.randint(0, len(self.training_points))  # random int intervalo [0, n + 1) => [0, n]

        # Actualizamos el valor del vector peso
        self.w += self.calculate_delta_weight(
            self.training_points[point],
            self.training_values[point],
            Perceptron.calculate_weighted_sum(self.training_points[point], self.w)
        )

        # Actualizamos el error
        self.error = self.calculate_error()
        if self.error < self.error_min:
            self.error_min = self.error
            self.w_min = np.copy(self.w)

        self.iteration += 1

    def train(self, status_callback: Optional[Callable[[np.ndarray], None]] = None) -> None:
        while not self.has_training_ended():
            self.do_training_iteration()
            if status_callback:
                status_callback(self.w)

    # Utiliza el w actual
    def _predict(self, point: np.ndarray, insert_identity_column: bool = False) -> float:
        if insert_identity_column:
            point = np.insert(point, 0, 1)
        return self.activation(Perceptron.calculate_weighted_sum(point, self.w))

    # Utiliza el w minimo
    def predict(self, point: np.ndarray, insert_identity_column: bool = False) -> float:
        if insert_identity_column:
            point = np.insert(point, 0, 1)
        return self.activation(Perceptron.calculate_weighted_sum(point, self.w_min))

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

        return np.array([point for idx, point in enumerate(points) if not math.isclose(self.predict(point), values[idx])])

    def is_validation_successful(self, points: np.ndarray, values: np.ndarray, insert_identity_column: bool = True) -> bool:
        if insert_identity_column:
            points = Perceptron.with_identity_dimension(points)

        for i in range(len(points)):
            if not math.isclose(self.predict(points[i]), values[i]):
                return False
        return True


# self.l_rate = params['learning_rate']
# self.activation_func = self.function_map[params['function']]
class SimplePerceptron(Perceptron):

    def __init__(self, l_rate: float, training_points: np.ndarray, training_values: np.ndarray, max_iteration: Optional[int] = None,
                 soft_reset_threshold: Optional[int] = None, insert_identity_column: bool = True) -> None:
        # Activation Function = funcion signo
        super().__init__(l_rate, np.sign, training_points, training_values, max_iteration, soft_reset_threshold, insert_identity_column)

    def calculate_delta_weight(self, point: np.ndarray, point_value: float, weighted_sum: float) -> np.ndarray:
        return self.l_rate * (point_value - self.activation(weighted_sum)) * point

    def calculate_error(self) -> float:
        return sum(abs(self.training_values[point] - self._predict(self.training_points[point])) for point in range(len(self.training_points)))


# linear
class LinearPerceptron(Perceptron):

    def __init__(self, l_rate: float, training_points: np.ndarray, training_values: np.ndarray, max_iteration: Optional[int] = None,
                 soft_reset_threshold: Optional[int] = None, insert_identity_column: bool = True) -> None:
        # Activation Function = funcion identidad
        super().__init__(l_rate, lambda x: x, training_points, training_values, max_iteration, soft_reset_threshold, insert_identity_column)

    def calculate_delta_weight(self, point: np.ndarray, point_value: float, weighted_sum: float) -> np.ndarray:
        return self.l_rate * (point_value - self.activation(weighted_sum)) * point

    def calculate_error(self) -> float:
        return super().calculate_error()


class NonLinearPerceptron(Perceptron):

    def __init__(self, l_rate: float, activation: ActivationFunction, activation_derivative: ActivationFunction,
                 training_points: np.ndarray, training_values: np.ndarray, max_iteration: Optional[int] = None,
                 soft_reset_threshold: Optional[int] = None, insert_identity_column: bool = True) -> None:

        super().__init__(l_rate, activation, training_points, training_values, max_iteration, soft_reset_threshold, insert_identity_column)
        self.activation_derivative = activation_derivative

    def calculate_delta_weight(self, point: np.ndarray, point_value: float, weighted_sum: float) -> np.ndarray:
        return self.l_rate * (point_value - self.activation(weighted_sum)) * point * self.activation_derivative(weighted_sum)

    def calculate_error(self) -> float:
        return super().calculate_error()
