import itertools
import math
from abc import ABC, abstractmethod
from typing import Callable, Optional, Tuple, List, Collection

import numpy as np

ActivationFunction = Callable[[float], float]


class Perceptron(ABC):

    DEFAULT_MAX_ITERATION: int = 10000
    DEFAULT_SOFT_RESET_THRESHOLD: int = 1000

    @staticmethod
    def calculate_weighted_sum(point: np.ndarray, w: np.array) -> float:
        return np.sum(point * w)

    @staticmethod
    def with_identity_dimension(points: np.ndarray) -> np.ndarray:
        return np.insert(points, 0, 1, axis=1)

    def __init__(self, l_rate: float, input_count: int, activation: ActivationFunction,
                 max_training_iteration: Optional[int] = None, soft_reset_threshold: Optional[int] = None) -> None:

        self.l_rate: float = l_rate
        self.input_count = input_count
        self.activation: ActivationFunction = activation
        self.error: Optional[float] = None

        # Training
        self.training_iteration: int = 0
        self.iters_since_soft_reset: int = 0
        self.max_training_iteration = (max_training_iteration if max_training_iteration is not None else Perceptron.DEFAULT_MAX_ITERATION)
        self.soft_reset_threshold = (soft_reset_threshold if soft_reset_threshold is not None else Perceptron.DEFAULT_SOFT_RESET_THRESHOLD)

    @abstractmethod
    def train(self, training_points: np.ndarray, training_values: np.ndarray,
              status_callback: Optional[Callable[[np.ndarray], None]] = None, insert_identity_column: bool = True) -> Tuple[int, np.ndarray]:
        pass

    @abstractmethod
    def predict(self, point: np.ndarray, insert_identity_column: bool = False) -> float:
        pass

    @abstractmethod
    def predict_points(self, points: np.ndarray, insert_identity_column: bool = True) -> np.ndarray:
        pass

    # Retorna los puntos que fueron predecidos incorrectamente
    @abstractmethod
    def validate_points(self, points: np.ndarray, values: np.ndarray, insert_identity_column: bool = True) -> np.ndarray:
        pass

    @abstractmethod
    def is_validation_successful(self, points: np.ndarray, values: np.ndarray, insert_identity_column: bool = True) -> bool:
        pass


class BaseSimplePerceptron(Perceptron):

    def __init__(self, l_rate: float, input_count: int, activation: ActivationFunction,
                 max_training_iteration: Optional[int] = None, soft_reset_threshold: Optional[int] = None, ) -> None:
        super().__init__(l_rate, input_count, activation, max_training_iteration, soft_reset_threshold)

        self.w: np.ndarray = np.random.uniform(-1, 1, input_count + 1)  # array de n + 1 puntos con dist. Uniforme([-1, 1))
        self.training_w: np.ndarray = np.copy(self.w)

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
        return self.error is not None and (math.isclose(self.error, 0) or self.training_iteration >= self.max_training_iteration)

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


class SimplePerceptron(BaseSimplePerceptron):

    def __init__(self, l_rate: float, input_count: int,
                 max_training_iteration: Optional[int] = None, soft_reset_threshold: Optional[int] = None) -> None:
        # Activation Function = funcion signo
        super().__init__(l_rate, input_count, np.sign, max_training_iteration, soft_reset_threshold)

    def calculate_delta_weight(self, point: np.ndarray, point_value: float, weighted_sum: float) -> np.ndarray:
        return self.l_rate * (point_value - self.activation(weighted_sum)) * point

    def calculate_error(self, training_points: np.ndarray, training_values: np.ndarray, w: np.ndarray) -> float:
        return sum(abs(training_values[point] - self._predict(training_points[point], w)) for point in range(len(training_points)))


class LinearPerceptron(BaseSimplePerceptron):

    def __init__(self, l_rate: float, input_count: int, max_training_iteration: Optional[int] = None,
                 soft_reset_threshold: Optional[int] = None) -> None:
        # Activation Function = funcion identidad
        super().__init__(l_rate, input_count, lambda x: x, max_training_iteration, soft_reset_threshold)

    def calculate_delta_weight(self, point: np.ndarray, point_value: float, weighted_sum: float) -> np.ndarray:
        return self.l_rate * (point_value - self.activation(weighted_sum)) * point

    def calculate_error(self, training_points: np.ndarray, training_values: np.ndarray, w: np.ndarray) -> float:
        return sum(0.5 * (training_values[point] - self._predict(training_points[point], w)) ** 2
                   for point in range(len(training_points)))


class NonLinearPerceptron(BaseSimplePerceptron):

    def __init__(self, l_rate: float, input_count: int, activation: ActivationFunction, activation_derivative: ActivationFunction,
                 max_training_iteration: Optional[int] = None, soft_reset_threshold: Optional[int] = None) -> None:
        super().__init__(l_rate, input_count, activation, max_training_iteration, soft_reset_threshold)
        self.activation_derivative = activation_derivative

    def calculate_delta_weight(self, point: np.ndarray, point_value: float, weighted_sum: float) -> np.ndarray:
        return self.l_rate * (point_value - self.activation(weighted_sum)) * point * self.activation_derivative(weighted_sum)

    def calculate_error(self, training_points: np.ndarray, training_values: np.ndarray, w: np.ndarray) -> float:
        return sum(0.5 * (training_values[point] - self._predict(training_points[point], w)) ** 2
                   for point in range(len(training_points)))


class PerceptronLayer:

    def __init__(self, size: int, perceptron_factory: Callable[[], NonLinearPerceptron]) -> None:
        self.size = size
        self.perceptrons: Collection[NonLinearPerceptron] = [perceptron_factory() for _ in range(self.size)]

    def predict(self, back_layer_prediction: np.ndarray) -> np.ndarray:
        return np.fromiter(
            # Concatenar el primer valor identidad (1) con las predicciones de los perceptrons
            itertools.chain(range(1, 1), (perceptron.predict(back_layer_prediction) for perceptron in self.perceptrons)),
            float
        )

    def calculate_error(self, front_layer_error: np.ndarray) -> np.ndarray:
        pass

class MultilayeredPerceptron(Perceptron):

    def __init__(self, l_rate: float, input_count: int, activation: ActivationFunction, activation_derivative: ActivationFunction,
                 layer_sizes: List[int], max_training_iteration: Optional[int] = None,
                 soft_reset_threshold: Optional[int] = None) -> None:

        super().__init__(l_rate, input_count, activation, max_training_iteration, soft_reset_threshold)
        self.activation_derivative: ActivationFunction = activation_derivative
        self.output_size = layer_sizes[-1]

        perceptron_factory: Callable[[], NonLinearPerceptron] =\
            lambda input_count: NonLinearPerceptron(
                l_rate, input_count, activation, activation_derivative,
                max_training_iteration, soft_reset_threshold
            )

        # layer_sizes[i + 1] = layer_size => layer_sizes[i] = perceptrons input size
        # Los perceptrones de la capa actual reciben una cantidad de inputs equivalente al tamaño de la layer anterior
        layer_sizes = [input_count] + layer_sizes # La primera capa recibe input_count inputs
        self.layers: List[PerceptronLayer] = [
            PerceptronLayer(layer_sizes[i + 1], lambda: perceptron_factory(layer_sizes[i])) for i in range(len(layer_sizes) - 1)
        ]