import itertools
import math
from abc import ABC, abstractmethod
from typing import Callable, Optional, List

import numpy as np

ActivationFunction = Callable[[float], float]

# class Perceptronn(ABC):
#     DEFAULT_MAX_ITERATION: int = 100
#     DEFAULT_SOFT_RESET_THRESHOLD: int = 1000
#
#     @staticmethod
#     def calculate_weighted_sum(point: np.ndarray, w: np.array) -> float:
#         return np.sum(point * w)
#
#     @staticmethod
#     def with_identity_dimension(points: np.ndarray) -> np.ndarray:
#         return np.insert(points, 0, 1, axis=1)
#
#     def __init__(self, l_rate: float, input_count: int, activation: ActivationFunction,
#                  max_training_iteration: Optional[int] = None, soft_reset_threshold: Optional[int] = None) -> None:
#         self.l_rate: float = l_rate
#         self.input_count = input_count
#         self.activation: ActivationFunction = activation
#         self.error: Optional[float] = None
#         self.excitement: Optional[float] = None
#
#         # Training
#         self.training_iteration: int = 0
#         self.iters_since_soft_reset: int = 0
#         self.max_training_iteration = (
#             max_training_iteration if max_training_iteration is not None else Perceptron.DEFAULT_MAX_ITERATION)
#         self.soft_reset_threshold = (
#             soft_reset_threshold if soft_reset_threshold is not None else Perceptron.DEFAULT_SOFT_RESET_THRESHOLD)
#
#     @abstractmethod
#     def train(self, training_points: np.ndarray, training_values: np.ndarray,
#               status_callback: Optional[Callable[[np.ndarray], None]] = None, insert_identity_column: bool = True) -> \
#             Tuple[int, np.ndarray]:
#         pass
#
#     @abstractmethod
#     def predict(self, point: np.ndarray, insert_identity_column: bool = False) -> float:
#         pass
#
#     @abstractmethod
#     def predict_points(self, points: np.ndarray, insert_identity_column: bool = True) -> np.ndarray:
#         pass
#
#     # Retorna los puntos que fueron predecidos incorrectamente
#     @abstractmethod
#     def validate_points(self, points: np.ndarray, values: np.ndarray,
#                         insert_identity_column: bool = True) -> np.ndarray:
#         pass
#
#     @abstractmethod
#     def is_validation_successful(self, points: np.ndarray, values: np.ndarray,
#                                  insert_identity_column: bool = True) -> bool:
#         pass
#
#
# class BaseSimplePerceptron(Perceptron):
#
#     def __init__(self, l_rate: float, input_count: int, activation: ActivationFunction,
#                  max_training_iteration: Optional[int] = None, soft_reset_threshold: Optional[int] = None, ) -> None:
#         super().__init__(l_rate, input_count, activation, max_training_iteration, soft_reset_threshold)
#
#         self.w: np.ndarray = np.random.uniform(-1, 1,
#                                                input_count + 1)  # array de n + 1 puntos con dist. Uniforme([-1, 1))
#         self.training_w: np.ndarray = np.copy(self.w)
#
#     @abstractmethod
#     def calculate_delta_weight(self, point: np.ndarray, point_value: float, weighted_sum: float) -> np.ndarray:
#         pass
#
#     @abstractmethod
#     def calculate_error(self, training_points: np.ndarray, training_values: np.ndarray, w: np.ndarray) -> float:
#         pass
#
#     def soft_training_reset(self) -> None:
#         self.training_w = np.random.uniform(-1, 1, len(self.training_w))
#         self.iters_since_soft_reset = 0
#
#     def hard_training_reset(self) -> None:
#         self.soft_training_reset()
#         self.training_iteration = 0
#
#     def has_training_ended(self) -> bool:
#         return self.error is not None and (
#                 math.isclose(self.error, 0) or self.training_iteration >= self.max_training_iteration)
#
#     def update_w(self, delta_w: np.ndarray):
#         self.training_w += delta_w
#
#     # No cambiar los training points en el medio del entrenamiento
#     # Antes de empezar un nuevo entrenamiento hacer un hard_training_reset
#     # Asume que los training_points ya tienen la columna identidad
#     def do_training_iteration(self, training_points: np.ndarray, training_values: np.ndarray) -> None:
#         if self.iters_since_soft_reset > self.soft_reset_threshold * len(training_points):
#             self.soft_training_reset()
#
#         # Seleccionamos un punto del training set al azar
#         point: int = np.random.randint(0, len(training_points))  # random int intervalo [0, n + 1) => [0, n]
#
#         # Actualizamos el valor del vector peso
#         self.update_w(self.calculate_delta_weight(
#             training_points[point],
#             training_values[point],
#             Perceptron.calculate_weighted_sum(training_points[point], self.training_w)
#         ))
#
#         # Actualizamos el error
#         current_error: float = self.calculate_error(training_points, training_values, self.training_w)
#         if self.error is None or current_error < self.error:
#             self.error = current_error
#             self.w = np.copy(self.training_w)
#
#         self.training_iteration += 1
#
#     def train(self, training_points: np.ndarray, training_values: np.ndarray,
#               status_callback: Optional[Callable[[np.ndarray], None]] = None, insert_identity_column: bool = True) -> \
#             Tuple[int, np.ndarray]:
#
#         if insert_identity_column:
#             training_points = Perceptron.with_identity_dimension(training_points)
#
#         while not self.has_training_ended():
#             self.do_training_iteration(training_points, training_values)
#             if status_callback:
#                 status_callback(self.training_w)
#
#         # Retorno al estado inicial y devuelvo el training_w final y la cantidad de training_iterations
#         ret: Tuple[int, np.ndarray] = (self.training_iteration, self.training_w)
#         self.hard_training_reset()
#
#         return ret
#
#     # Asume que el punto tiene la columna identidad
#     def _predict(self, point: np.ndarray, w: np.ndarray) -> float:
#         return self.activation(Perceptron.calculate_weighted_sum(point, w))
#
#     def predict(self, point: np.ndarray, insert_identity_column: bool = False) -> float:
#         if insert_identity_column:
#             point = np.insert(point, 0, 1)
#         self.excitement = Perceptron.calculate_weighted_sum(point, self.w)
#         return self.activation(self.excitement)
#
#     def training_predict(self, point: np.ndarray, insert_identity_column: bool = False) -> float:
#         if insert_identity_column:
#             point = np.insert(point, 0, 1)
#         self.excitement = Perceptron.calculate_weighted_sum(point, self.training_w)
#         return self.activation(self.excitement)
#
#     # TODO(tobi): Alguno sabe una mejor manera? - Es mejor con fromiter o array
#     def predict_points(self, points: np.ndarray, insert_identity_column: bool = True) -> np.ndarray:
#         if insert_identity_column:
#             points = Perceptron.with_identity_dimension(points)
#         return np.array(map(self.predict, points))
#
#     # Retorna los puntos que fueron predecidos incorrectamente
#     # TODO(tobi): Alguno sabe una mejor manera? - Es mejor con fromiter o array
#     def validate_points(self, points: np.ndarray, values: np.ndarray,
#                         insert_identity_column: bool = True) -> np.ndarray:
#         if insert_identity_column:
#             points = Perceptron.with_identity_dimension(points)
#
#         return np.array([points[point] for point in range(len(points)) if
#                          not math.isclose(self.predict(points[point]), values[point])])
#
#     def is_validation_successful(self, points: np.ndarray, values: np.ndarray,
#                                  insert_identity_column: bool = True) -> bool:
#         if insert_identity_column:
#             points = Perceptron.with_identity_dimension(points)
#
#         for i in range(len(points)):
#             if not math.isclose(self.predict(points[i]), values[i]):
#                 return False
#         return True
#
#
# class SimplePerceptron(BaseSimplePerceptron):
#
#     def __init__(self, l_rate: float, input_count: int,
#                  max_training_iteration: Optional[int] = None, soft_reset_threshold: Optional[int] = None) -> None:
#         # Activation Function = funcion signo
#         super().__init__(l_rate, input_count, np.sign, max_training_iteration, soft_reset_threshold)
#
#     def calculate_delta_weight(self, point: np.ndarray, point_value: float, weighted_sum: float) -> np.ndarray:
#         return self.l_rate * (point_value - self.activation(weighted_sum)) * point
#
#     def calculate_error(self, training_points: np.ndarray, training_values: np.ndarray, w: np.ndarray) -> float:
#         return sum(abs(training_values[point] - self._predict(training_points[point], w)) for point in
#                    range(len(training_points)))
#
#
# class LinearPerceptron(BaseSimplePerceptron):
#
#     def __init__(self, l_rate: float, input_count: int, max_training_iteration: Optional[int] = None,
#                  soft_reset_threshold: Optional[int] = None) -> None:
#         # Activation Function = funcion identidad
#         super().__init__(l_rate, input_count, lambda x: x, max_training_iteration, soft_reset_threshold)
#
#     def calculate_delta_weight(self, point: np.ndarray, point_value: float, weighted_sum: float) -> np.ndarray:
#         return self.l_rate * (point_value - self.activation(weighted_sum)) * point
#
#     def calculate_error(self, training_points: np.ndarray, training_values: np.ndarray, w: np.ndarray) -> float:
#         return sum(0.5 * (training_values[point] - self._predict(training_points[point], w)) ** 2
#                    for point in range(len(training_points)))
#
#
# class NonLinearPerceptron(BaseSimplePerceptron):
#
#     def __init__(self, l_rate: float, input_count: int, activation: ActivationFunction,
#                  activation_derivative: ActivationFunction,
#                  max_training_iteration: Optional[int] = None, soft_reset_threshold: Optional[int] = None) -> None:
#         super().__init__(l_rate, input_count, activation, max_training_iteration, soft_reset_threshold)
#         self.activation_derivative = activation_derivative
#
#     def calculate_delta_weight(self, point: np.ndarray, point_value: float, weighted_sum: float) -> np.ndarray:
#         return self.l_rate * (point_value - self.activation(weighted_sum)) * point * self.activation_derivative(
#             weighted_sum)
#
#     def calculate_error(self, training_points: np.ndarray, training_values: np.ndarray, w: np.ndarray) -> float:
#         return sum(0.5 * (training_values[point] - self._predict(training_points[point], w)) ** 2
#                    for point in range(len(training_points)))
#
#
# class PerceptronnLayer:
#
#     def __init__(self, size: int, perceptron_factory: Callable[[], NonLinearPerceptron]) -> None:
#         self.size = size
#         self.perceptrons: Collection[NonLinearPerceptron] = [perceptron_factory() for _ in range(self.size)]
#         self.excitements = None
#         self.activation = None
#         self.delta = None
#
#     def predict(self, back_layer_prediction: np.ndarray) -> None:
#         self.activation = np.fromiter(
#             # Concatenar el primer valor identidad (1) con las predicciones de los perceptrons
#             itertools.chain(range(1, 2),
#                             (perceptron.predict(back_layer_prediction) for perceptron in self.perceptrons)),
#             float
#         )
#         self._get_excitements()
#
#     def training_predict(self, back_layer_prediction: np.ndarray) -> None:
#         self.activation = np.fromiter(
#             # Concatenar el primer valor identidad (1) con las predicciones de los perceptrons
#             itertools.chain(range(1, 2),
#                             (perceptron.training_predict(back_layer_prediction) for perceptron in self.perceptrons)),
#             float
#         )
#         self._get_excitements()
#
#     def _get_excitements(self) -> None:
#         self.excitements = np.fromiter(
#             # Concatenar el primer valor identidad (1) con las predicciones de los perceptrons
#             itertools.chain(range(1, 2), (perceptron.excitement for perceptron in self.perceptrons)),
#             float
#         )
#
#     def calculate_previous_delta(self, previous_excitements: np.ndarray,
#                                  activation_derivative: ActivationFunction) -> np.ndarray:
#
#         # Hacer la suma pesada entre el componente del delta del perceptron correspondiente y el peso para cada uno de los nodos de la capa de abajo
#         weighted_sums: List[float] = []
#
#         # No interesa el primer peso porque es el ficticio
#         for j in range(1, len(self.perceptrons[0].training_w)):
#             weighted_sum: float = 0
#             for i, perceptron in enumerate(self.perceptrons):
#                 weighted_sum += self.delta[i] * perceptron.training_w[j]
#             weighted_sums.append(weighted_sum)
#         # aca hay que ignorar la primer excitacion porque tambien es la ficticia
#         return np.fromiter(
#             (weighted_sums[i - 1] * activation_derivative(previous_excitements[i])
#              for i in range(1, len(previous_excitements))),
#             float
#         )
#
#     def update_w(self, l_rate: float, previous_activations: np.ndarray) -> None:
#         for i, perceptron in enumerate(self.perceptrons):
#             perceptron.update_w(l_rate * self.delta[i] * previous_activations)
#
#
# class MultilayeredPerceptron(Perceptron):
#
#     def predict(self, point: np.ndarray, insert_identity_column: bool = False) -> float:
#         pass
#
#     def predict_points(self, points: np.ndarray, insert_identity_column: bool = True) -> np.ndarray:
#         pass
#
#     def validate_points(self, points: np.ndarray, values: np.ndarray,
#                         insert_identity_column: bool = True) -> np.ndarray:
#         pass
#
#     def is_validation_successful(self, points: np.ndarray, values: np.ndarray,
#                                  insert_identity_column: bool = True) -> bool:
#         pass
#
#     def __init__(self, l_rate: float, input_count: int, activation: ActivationFunction,
#                  activation_derivative: ActivationFunction,
#                  layer_sizes: List[int], max_training_iteration: Optional[int] = None,
#                  soft_reset_threshold: Optional[int] = None) -> None:
#
#         super().__init__(l_rate, input_count, activation, max_training_iteration, soft_reset_threshold)
#         self.activation_derivative: ActivationFunction = activation_derivative
#         self.output_size = layer_sizes[-1]
#
#         perceptron_factory: Callable[[], NonLinearPerceptron] = \
#             lambda input_count: NonLinearPerceptron(
#                 l_rate, input_count, activation, activation_derivative,
#                 max_training_iteration, soft_reset_threshold
#             )
#
#         # layer_sizes[i + 1] = layer_size => layer_sizes[i] = perceptrons input size
#         # Los perceptrones de la capa actual reciben una cantidad de inputs equivalente al tamaño de la layer anterior
#         layer_sizes = [input_count] + layer_sizes  # La primera capa recibe input_count inputs
#         self.layers: List[PerceptronLayer] = [
#             PerceptronLayer(layer_sizes[i + 1], lambda: perceptron_factory(layer_sizes[i])) for i in
#             range(len(layer_sizes) - 1)
#         ]
#
#     # Se recibe xi supra mu
#     def _training_predict(self, training_points: np.ndarray) -> np.ndarray:
#
#         self.layers[0].training_predict(training_points)
#
#         # Conseguir el V de la capa de salida, propagando los puntos de entrada por todas las capas
#         for m in range(1, len(self.layers)):
#             self.layers[m].training_predict(self.layers[m - 1].activation)
#
#         return self.layers[-1].activation
#
#     def get_first_delta(self, training_value: np.ndarray) -> np.ndarray:
#         return np.array(
#             [self.activation_derivative(perceptron.excitement) * (training_value[i] - self.layers[-1].activation[i + 1])
#              for i, perceptron in enumerate(self.layers[-1].perceptrons)])
#
#     def calculate_error(self, training_points: np.ndarray, training_values: np.ndarray) -> float:
#
#         return np.fromiter((0.5 * (training_values[i] - self._training_predict(training_points[i])[1:]) ** 2 for i in
#                             range(len(training_points))), float).sum()
#
#     def do_training_iteration(self, training_points: np.ndarray, training_values: np.ndarray) -> None:
#
#         point: int = np.random.randint(0, len(training_points))
#
#         self._training_predict(training_points[point])
#
#         self.layers[-1].delta = self.get_first_delta(training_values[point])
#
#         # Capas M-2 -> 0
#         for m in range(len(self.layers) - 2, -1, -1):
#             self.layers[m].delta = self.layers[m + 1].calculate_previous_delta(
#                 self.layers[m].excitements, self.activation_derivative)
#
#         self.layers[0].update_w(self.l_rate, training_points[point])
#
#         for m in range(1, len(self.layers)):
#             self.layers[m].update_w(self.l_rate, self.layers[m - 1].activation)
#
#         self.error = self.calculate_error(training_points, training_values)
#
#     def has_training_ended(self) -> bool:
#         return self.error is not None and (
#                 math.isclose(self.error, 0) or self.training_iteration >= self.max_training_iteration)
#
#     def train(self, training_points: np.ndarray, training_values: np.ndarray,
#               status_callback: Optional[Callable[[np.ndarray], None]] = None,
#               insert_identity_column: bool = True) -> None:
#
#         if insert_identity_column:
#             training_points = Perceptron.with_identity_dimension(training_points)
#
#         if len(np.shape(training_values)) == 1:
#             training_values = np.reshape(training_values, (training_values.size, 1))
#
#         while not self.has_training_ended():
#             self.do_training_iteration(training_points, training_values)
#             print(self.error)
#             # if status_callback:
#             #     status_callback(self.training_w)
#
#     # TODO Capaz hay que implementar el hard reset??


# ML.predict
# ML.predict_point
# ML.train

class NeuralNetwork(ABC):
    DEFAULT_VARIABLE_LEARNING_RATE_FACTOR = 1
    DEFAULT_MAX_ITERATION: int = 10000000
    DEFAULT_SOFT_RESET_THRESHOLD: int = 10000000

    def __init__(self, max_training_iteration=None) -> None:
        super().__init__()
        self.max_training_iteration = (
            max_training_iteration if max_training_iteration is not None else NeuralNetwork.DEFAULT_MAX_ITERATION)
        self.error: float = float("inf")
        self.training_iteration: int = 0

    @staticmethod
    def with_identity_dimension(points: np.ndarray) -> np.ndarray:
        return np.insert(points, 0, 1, axis=1)

    @abstractmethod
    def predict(self, points: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def predict_points(self, points: np.ndarray) -> np.ndarray:
        pass

    def has_training_ended(self) -> bool:
        return self.error is not None and (
                math.isclose(self.error, 0) or self.training_iteration >= self.max_training_iteration)

    def train(self, training_points: np.ndarray, training_values: np.ndarray) -> None:
        training_points = NeuralNetwork.with_identity_dimension(training_points)

        while not self.has_training_ended():
            self.do_training_iteration(training_points, training_values)

    @abstractmethod
    def do_training_iteration(self, training_points: np.ndarray, training_values: np.ndarray) -> None:
        pass

    @abstractmethod
    def calculate_error(self, training_points: np.ndarray, training_values: np.ndarray) -> float:
        pass


class Perceptron:

    def __init__(self, l_rate: float, input_count: int, activation_fn: ActivationFunction, momentum_factor: float):
        self.activation_fn: ActivationFunction = activation_fn
        self.l_rate: float = l_rate
        self.input_count = input_count
        self.excitement: float = 0
        self.activation: float = 0
        self.delta: float = 0
        self.w: np.ndarray = np.zeros(input_count + 1)
        self.training_w: np.ndarray = np.zeros(input_count + 1)
        self.momentum_factor: float = momentum_factor
        self.previous_delta_w: np.ndarray = np.zeros(input_count + 1)

        self.training_weights_reset()
        self.persist_weights()

    def predict(self, points: np.ndarray) -> float:
        return self._predict(points, self.w)

    def training_predict(self, points: np.ndarray) -> float:
        return self._predict(points, self.training_w)

    def _predict(self, points: np.ndarray, weights: np.ndarray) -> float:
        self.excitement = sum(points * weights)
        self.activation = self.activation_fn(self.excitement)
        return self.activation

    def persist_weights(self) -> None:
        self.w = self.training_w

    def update_training_weights(self, inputs: np.ndarray) -> None:
        new_delta_w: np.ndarray = self.l_rate * self.delta * inputs + self.momentum_factor * self.previous_delta_w
        self.training_w += new_delta_w
        self.previous_delta_w = new_delta_w

    def training_weights_reset(self) -> None:
        self.training_w = np.random.uniform(-1, 1, self.input_count + 1)

    def update_eta(self, delta_l_rate: float) -> None:
        self.l_rate += delta_l_rate


class BaseSinglePerceptronNeuralNetwork(NeuralNetwork):

    def __init__(self,
                 l_rate: float,
                 input_count: int,
                 activation_fn: ActivationFunction,
                 momentum_factor: float = 0,
                 max_training_iteration: Optional[int] = None,
                 soft_reset_threshold: Optional[int] = None,
                 variable_learning_rate_factor: Optional[float] = None) -> None:

        super().__init__(max_training_iteration)

        self.l_rate: float = l_rate
        self.input_count = input_count
        self.activation_fn: ActivationFunction = activation_fn
        self.momentum_factor = momentum_factor

        self.perceptron = Perceptron(l_rate, input_count, activation_fn)

        self.variable_learning_rate_factor: float = (
            variable_learning_rate_factor if variable_learning_rate_factor is not None else NeuralNetwork.DEFAULT_VARIABLE_LEARNING_RATE_FACTOR)

        # Training
        self.concurrent_error_improvements: int = 0
        self.concurrent_error_deterioration: int = 0

        self.iters_since_soft_reset: int = 0
        self.soft_reset_threshold = (
            soft_reset_threshold if soft_reset_threshold is not None else Perceptron.DEFAULT_SOFT_RESET_THRESHOLD)

    def predict(self, points: np.ndarray) -> np.ndarray:
        return np.array(self.perceptron.predict(points))

    def predict_points(self, points: np.ndarray) -> np.ndarray:
        return np.apply_along_axis(self.perceptron.predict, 1, points)

    def train(self, training_points: np.ndarray, training_values: np.ndarray) -> None:
        training_values = np.squeeze(training_values)
        super().train(training_points, training_values)

    # Training values es boxed o no?
    def do_training_iteration(self, training_points: np.ndarray, training_values: np.ndarray) -> None:
        if self.iters_since_soft_reset > self.soft_reset_threshold * len(training_points):
            self.perceptron.training_weights_reset()

        # Seleccionamos un punto del training set al azar
        point: int = np.random.randint(0, len(training_points))  # random int intervalo [0, n + 1) => [0, n]

        # Actualizamos el valor del vector peso
        self.perceptron.delta = self.calculate_delta(training_values[point])

        self.perceptron.update_training_weights(training_points[point])

        # Actualizamos el error
        current_error: float = self.calculate_error(training_points, training_values)

        if self.error is None or current_error < self.error:
            self.error = current_error
            self.perceptron.persist_weights()
            self._update_eta(True)
        else:
            self._update_eta(False)

        self.training_iteration += 1

    def _update_eta(self, error_updated: bool) -> None:
        if error_updated:
            self.concurrent_error_improvements += 1
            self.concurrent_error_deterioration = 0
            if self.concurrent_error_improvements == 5:
                self.perceptron.update_eta()  # config

        else:
            self.concurrent_error_deterioration += 1
            self.concurrent_error_improvements = 0
            if self.concurrent_error_deterioration == 10:
                self.perceptron.update_eta()  # config

    def calculate_error(self, training_points: np.ndarray, training_values: np.ndarray) -> float:

        # FIXME Puede hacer falta que sea una lambda
        predictions: np.ndarray = np.apply_along_axis(self.perceptron.training_predict, 1, training_points)
        return sum(abs(training_points - predictions))

    def calculate_delta(self, training_values: np.ndarray) -> float:
        return training_values - self.perceptron.activation


class StepNeuralNetwork(BaseSinglePerceptronNeuralNetwork):

    def __init__(self,
                 l_rate: float,
                 input_count: int,
                 momentum_factor: float = 0,
                 max_training_iteration: Optional[int] = None,
                 soft_reset_threshold: Optional[int] = None,
                 variable_learning_rate_factor: Optional[float] = None) -> None:
        super().__init__(l_rate, input_count, np.sign, momentum_factor, max_training_iteration, soft_reset_threshold, variable_learning_rate_factor)


class LinearNeuralNetwork(BaseSinglePerceptronNeuralNetwork):

    def __init__(self,
                 l_rate: float,
                 input_count: int,
                 momentum_factor: float = 0,
                 max_training_iteration: Optional[int] = None,
                 soft_reset_threshold: Optional[int] = None,
                 variable_learning_rate_factor: Optional[float] = None) -> None:
        super().__init__(l_rate, input_count, lambda x: x, momentum_factor, max_training_iteration, soft_reset_threshold, variable_learning_rate_factor)

    def calculate_error(self, training_points: np.ndarray, training_values: np.ndarray) -> float:
        predictions: np.ndarray = np.apply_along_axis(self.perceptron.training_predict, 1, training_points)
        return sum(0.5 * (training_points - predictions) ** 2)


class NonLinearSinglePerceptronNeuralNetwork(BaseSinglePerceptronNeuralNetwork):

    def __init__(self,
                 l_rate: float,
                 input_count: int,
                 activation_fn: ActivationFunction,
                 activation_derivative: ActivationFunction,
                 momentum_factor: float = 0,
                 max_training_iteration: Optional[int] = None,
                 soft_reset_threshold: Optional[int] = None,
                 variable_learning_rate_factor: Optional[float] = None) -> None:
        super().__init__(l_rate, input_count, activation_fn, momentum_factor, max_training_iteration, soft_reset_threshold, variable_learning_rate_factor)
        self.activation_derivative: ActivationFunction = activation_derivative

    def calculate_error(self, training_points: np.ndarray, training_values: np.ndarray) -> float:
        predictions: np.ndarray = np.apply_along_axis(self.perceptron.predict, 1, training_points)
        return sum(0.5 * (training_points - predictions) ** 2)

    def calculate_delta(self, training_values: np.ndarray) -> float:
        return (training_values - self.perceptron.activation) * self.activation_derivative(self.perceptron.excitement)


class PerceptronLayer:

    def __init__(self, size: int, perceptron_factory: Callable[[], Perceptron],
                 activation_derivative: ActivationFunction) -> None:
        self.size = size
        self.perceptrons: List[Perceptron] = [perceptron_factory() for _ in range(self.size)]
        self.activation_derivative: ActivationFunction = activation_derivative
        self.activation_cache: np.ndarray = np.zeros(self.size + 1)

    def training_predict(self, points: np.ndarray) -> np.ndarray:

        self.activation_cache = np.fromiter(
            # Concatenar el primer valor identidad (1) con las predicciones de los perceptrons
            itertools.chain(range(1, 2),
                            (perceptron.training_predict(points) for perceptron in self.perceptrons)),
            float
        )

        return self.activation_cache

    def predict(self, points: np.ndarray) -> np.ndarray:

        self.activation_cache = np.fromiter(
            # Concatenar el primer valor identidad (1) con las predicciones de los perceptrons
            itertools.chain(range(1, 2),
                            (perceptron.predict(points) for perceptron in self.perceptrons)),
            float
        )

        return self.activation_cache

    def update_perceptrons_delta(self, delta_multiplier: np.ndarray) -> None:
        for i, perceptron in enumerate(self.perceptrons):
            perceptron.delta = self.activation_derivative(perceptron.excitement) * delta_multiplier[i]

    def calculate_previous_layer_delta_multiplier(self) -> np.ndarray:

        # Hacer la suma pesada entre el componente del delta del perceptron correspondiente y el peso para cada uno de los nodos de la capa de abajo
        weighted_sums: List[float] = []

        # No interesa el primer peso porque es el ficticio
        for j in range(1, len(self.perceptrons[0].training_w)):
            weighted_sum: float = 0

            for perceptron in self.perceptrons:
                weighted_sum += perceptron.delta * perceptron.training_w[j]

            weighted_sums.append(weighted_sum)

        # aca hay que ignorar la primer excitacion porque tambien es la ficticia
        return np.array(weighted_sums)

    def update_w(self, previous_activations: np.ndarray):
        for perceptron in self.perceptrons:
            perceptron.update_training_weights(previous_activations)

    def persist_training_weights(self) -> None:
        for perceptron in self.perceptrons:
            perceptron.persist_weights()

    def training_weights_reset(self) -> None:
        for perceptron in self.perceptrons:
            perceptron.training_weights_reset()

    def update_eta(self, delta_l_rate: float) -> None:
        for perceptron in self.perceptrons:
            perceptron.update_eta(delta_l_rate)


class MultilayeredNeuralNetwork(NeuralNetwork):

    def __init__(self,
                 l_rate: float,
                 input_count: int,
                 activation_fn: ActivationFunction,
                 activation_derivative: ActivationFunction,
                 layer_sizes: List[int],
                 momentum_factor: float = 0,
                 max_training_iteration: Optional[int] = None,
                 soft_reset_threshold: Optional[int] = None, variable_learning_rate_factor: Optional[float] = None) -> None:

        super().__init__(max_training_iteration)

        self.l_rate: float = l_rate
        self.input_count = input_count
        self.activation_fn: ActivationFunction = activation_fn
        self.activation_derivative: ActivationFunction = activation_derivative
        self.variable_learning_rate_factor: float = (
            variable_learning_rate_factor if variable_learning_rate_factor is not None else NeuralNetwork.DEFAULT_VARIABLE_LEARNING_RATE_FACTOR)

        # Training
        self.concurrent_error_improvements: int = 0
        self.concurrent_error_deterioration: int = 0

        self.iters_since_soft_reset: int = 0
        self.soft_reset_threshold = (
            soft_reset_threshold if soft_reset_threshold is not None else NeuralNetwork.DEFAULT_SOFT_RESET_THRESHOLD)

        perceptron_factory: Callable[[], Perceptron] = \
            lambda input_count: Perceptron(
                l_rate, input_count, activation_fn, momentum_factor
            )

        # layer_sizes[i + 1] = layer_size => layer_sizes[i] = perceptrons input size
        # Los perceptrones de la capa actual reciben una cantidad de inputs equivalente al tamaño de la layer anterior
        layer_sizes = [input_count] + layer_sizes  # La primera capa recibe input_count inputs
        self.layers: List[PerceptronLayer] = [
            PerceptronLayer(layer_sizes[i + 1], lambda: perceptron_factory(layer_sizes[i]), activation_derivative) for i
            in
            range(len(layer_sizes) - 1)
        ]

    def predict(self, points: np.ndarray) -> np.ndarray:
        last_prediction: np.ndarray = points

        # Conseguir el V de la capa de salida, propagando los puntos de entrada por todas las capas
        for m in range(len(self.layers)):
            last_prediction = self.layers[m].predict(last_prediction)

        return last_prediction

    def predict_points(self, points: np.ndarray) -> np.ndarray:
        pass

    def _has_training_ended(self) -> bool:
        return self.error is not None and (
                math.isclose(self.error, 0) or self.training_iteration >= self.max_training_iteration)

    def do_training_iteration(self, training_points: np.ndarray, training_values: np.ndarray) -> None:

        if self.iters_since_soft_reset > self.soft_reset_threshold * len(training_points):
            self._training_weights_reset()

        point: int = np.random.randint(0, len(training_points))

        prediction: np.ndarray = self._training_predict(training_points[point])

        self._update_deltas(training_values[point], prediction)

        self._update_weights(training_points[point])

        current_error: float = self.calculate_error(training_points, training_values)

        if current_error < self.error:
            self.error = current_error
            self._persist_training_weights()
            self._update_eta(True)
            print(self.error, self.training_iteration)
        else:
            self._update_eta(False)

        self.training_iteration += 1

    def _update_eta(self, error_updated: bool) -> None:
        if error_updated:
            self.concurrent_error_improvements += 1
            self.concurrent_error_deterioration = 0
            if self.concurrent_error_improvements == 5:
                self._persist_updated_eta(self.variable_learning_rate_factor)  # config

        else:
            self.concurrent_error_deterioration += 1
            self.concurrent_error_improvements = 0
            if self.concurrent_error_deterioration == 10:
                self._persist_updated_eta(- self.variable_learning_rate_factor / 4 * self.l_rate)  # config

    def _persist_updated_eta(self, delta_learning_rate: float):
        self.l_rate += delta_learning_rate
        for m in range(len(self.layers)):
            self.layers[m].update_eta(delta_learning_rate)

    def calculate_error(self, training_points: np.ndarray, training_values: np.ndarray) -> float:

        predictions: np.ndarray = np.apply_along_axis(self._training_predict, 1, training_points)[:, 1:]
        return 0.5 * sum((predictions.flatten() - training_values.flatten()) ** 2)

    def _training_predict(self, training_points: np.ndarray) -> np.ndarray:

        last_prediction: np.ndarray = training_points

        # Conseguir el V de la capa de salida, propagando los puntos de entrada por todas las capas
        for m in range(len(self.layers)):
            last_prediction = self.layers[m].training_predict(last_prediction)

        return last_prediction

    def _update_deltas(self, training_values: np.ndarray, prediction: np.ndarray) -> None:

        delta_multiplier: np.ndarray = training_values - prediction[1:]

        for m in range(len(self.layers) - 1, -1, -1):
            self.layers[m].update_perceptrons_delta(delta_multiplier)
            delta_multiplier = self.layers[m].calculate_previous_layer_delta_multiplier()

    def _update_weights(self, training_points: np.ndarray) -> None:

        previous_activation: np.ndarray = training_points

        for m in range(len(self.layers)):
            self.layers[m].update_w(previous_activation)
            previous_activation = self.layers[m].activation_cache

    def _persist_training_weights(self):
        for m in range(len(self.layers)):
            self.layers[m].persist_training_weights()

    def _training_weights_reset(self):
        for m in range(len(self.layers)):
            self.layers[m].training_weights_reset()
