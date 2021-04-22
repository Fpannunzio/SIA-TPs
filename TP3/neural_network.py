import functools
import itertools
import math
from abc import ABC, abstractmethod
from enum import Enum

import attr
from typing import Callable, Optional, List, Union, Any

import numpy as np
from scipy.optimize import minimize_scalar, OptimizeResult

ActivationFunction = Callable[[float], float]
_ErrorFunction = Callable[[np.ndarray, np.ndarray], float]
_ErrorFactorCalculator = Callable[[float, float], float]

DEFAULT_MAX_ITERATIONS: int = 100
DEFAULT_ERROR_TOLERANCE: float = 1e-9
DEFAULT_L_RATE_LINEAR_SEARCH_MAX_ITERATIONS: int = 500
DEFAULT_L_RATE_LINEAR_SEARCH_ERROR_TOLERANCE: float = 1e-5


def assert_not_none(obj: Optional[Any]) -> Any:
    if obj is None:
        raise TypeError()
    return obj


@attr.s(auto_attribs=True)
class _ErrorFunctionContainer:
    error_function: _ErrorFunction
    error_factor_calculator: _ErrorFactorCalculator

    def calculate_error(self, values: np.ndarray, activations: np.ndarray) -> float:
        return self.error_function(values, activations)

    def calculate_error_factor(self, value: float, activation: float) -> float:
        return self.error_factor_calculator(value, activation)


class NeuralNetworkErrorFunction(Enum):
    ABSOLUTE = _ErrorFunctionContainer(
        lambda values, activations: sum(abs(values - activations)),
        lambda value, activation: value - activation
    )

    QUADRATIC = _ErrorFunctionContainer(
        lambda values, activations: sum((values - activations) ** 2) / 2,
        lambda value, activation: value - activation
    )

    LOGARITHMIC = _ErrorFunctionContainer(
        lambda values, activations:  # Numpy Magic
        sum((1 + values) * np.log((1 + values)/(1 + activations)) + (1 - values) * np.log((1 - values)/(1 - activations))) / 2,

        lambda value, activation: (value - activation)/(1 - activation ** 2)
    )

    def calculate_error(self, values: np.ndarray, activations: np.ndarray) -> float:
        return self.value.calculate_error(values, activations)

    def calculate_error_factor(self, value: float, activation: float) -> float:
        return self.value.calculate_error_factor(value, activation)


@attr.s(auto_attribs=True)
class NeuralNetworkBaseConfiguration:
    input_count: Optional[int] = None  # Required
    output_count: Optional[int] = None  # Required
    activation_fn: Optional[ActivationFunction] = None  # Required
    error_function: Optional[NeuralNetworkErrorFunction] = None  # Required
    max_training_iterations: int = DEFAULT_MAX_ITERATIONS
    soft_reset_threshold: Optional[int] = None  # max_training_iterations
    max_stale_error_iterations: Optional[int] = None  # max_training_iterations
    training_error_goal: float = 0
    training_error_tolerance: float = DEFAULT_ERROR_TOLERANCE
    momentum_factor: float = 0
    linear_search_l_rate: bool = False
    linear_search_l_rate_max_iterations: int = DEFAULT_L_RATE_LINEAR_SEARCH_MAX_ITERATIONS
    linear_search_l_rate_error_tolerance: float = DEFAULT_L_RATE_LINEAR_SEARCH_ERROR_TOLERANCE
    base_l_rate: Optional[float] = None  # Required
    l_rate_up_scaling_factor: float = 0
    l_rate_down_scaling_factor: float = 0
    error_positive_trend_threshold: Optional[int] = None  # max_training_iterations
    error_negative_trend_threshold: Optional[int] = None  # max_training_iterations

    def set_runtime_defaults(self):
        # Runtime Defaults
        if self.soft_reset_threshold is None          : self.soft_reset_threshold           = self.max_training_iterations
        if self.max_stale_error_iterations is None    : self.max_stale_error_iterations     = self.max_training_iterations
        if self.error_positive_trend_threshold is None: self.error_positive_trend_threshold = self.max_training_iterations
        if self.error_negative_trend_threshold is None: self.error_negative_trend_threshold = self.max_training_iterations

    def valid_or_fail(self) -> None:
        valid: bool = (
            (self.input_count is not None and self.input_count > 0) and
            (self.output_count is not None and self.output_count > 0) and
            self.activation_fn is not None and
            self.error_function is not None and
            self.max_training_iterations > 0 and
            self.soft_reset_threshold > 0 and
            self.max_stale_error_iterations > 0 and
            self.training_error_goal > 0 and
            self.training_error_tolerance > 0 and
            self.momentum_factor >= 0 and
            (self.linear_search_l_rate or self.base_l_rate is not None) and  # Base l_rate or linear search
            self.linear_search_l_rate_max_iterations > 0 and
            self.linear_search_l_rate_error_tolerance > 0 and
            (self.base_l_rate is None or self.base_l_rate > 0) and
            self.l_rate_up_scaling_factor >= 0 and
            self.l_rate_down_scaling_factor >= 0 and
            self.error_positive_trend_threshold > 0 and
            self.error_negative_trend_threshold > 0
        )

        if not valid:
            raise ValueError(f'Invalid Neural Network base configuration:\n{repr(self)}')


class NeuralNetwork(ABC):

    @staticmethod
    def with_identity_dimension(points: np.ndarray, axis: int = 1) -> np.ndarray:
        return np.insert(points, 0, 1, axis=axis)

    def __init__(self, base_config: NeuralNetworkBaseConfiguration) -> None:

        base_config.set_runtime_defaults()
        base_config.valid_or_fail()
        self.input_count: int = assert_not_none(base_config.input_count)
        self.output_count: int = assert_not_none(base_config.output_count)
        self.activation_fn: ActivationFunction = assert_not_none(base_config.activation_fn)
        self.error_function: NeuralNetworkErrorFunction = assert_not_none(base_config.error_function)
        self.max_training_iterations: int = base_config.max_training_iterations
        self.soft_reset_threshold: int = base_config.soft_reset_threshold
        self.max_stale_error_iterations: int = base_config.max_stale_error_iterations
        self.training_error_goal: float = base_config.training_error_goal
        self.training_error_tolerance: float = base_config.training_error_tolerance
        self.momentum_factor: float = base_config.momentum_factor
        self.linear_search_l_rate: bool = base_config.linear_search_l_rate
        self.linear_search_l_rate_max_iterations: int = base_config.linear_search_l_rate_max_iterations
        self.linear_search_l_rate_error_tolerance: float = base_config.linear_search_l_rate_error_tolerance
        self.l_rate: float = (base_config.base_l_rate if base_config.base_l_rate is not None else -1)
        self.l_rate_up_scaling_factor: float = base_config.l_rate_up_scaling_factor
        self.l_rate_down_scaling_factor: float = base_config.l_rate_down_scaling_factor
        self.error_positive_trend_threshold: float = base_config.error_positive_trend_threshold
        self.error_negative_trend_threshold: float = base_config.error_negative_trend_threshold

        self.error: float = np.inf
        self.last_training_error: float = self.error
        self.training_iteration: int = 0
        self.iters_since_soft_reset: int = 0
        self.stale_error_count: int = 0

        # Momentum
        self.error_positive_trend: int = 0
        self.error_negative_trend: int = 0

    def has_training_ended(self) -> bool:
        return (
            self.stale_error_count > self.max_stale_error_iterations or
            math.isclose(self.error, 0, abs_tol=self.training_error_goal + self.training_error_tolerance) or
            self.training_iteration >= self.max_training_iterations
        )

    def _validate_training_data(self, training_points: np.ndarray, training_values: np.ndarray) -> None:
        valid: bool = (
            len(training_points) > 0 and
            len(training_points) == len(training_values) and
            len(training_points[0]) == self.input_count and
            # Tamanio correcto de values, tomando en cuenta que puede ser un array
            ((self.output_count == 1 and len(training_values.shape) == 1) or len(training_values[0]) == self.output_count)
        )
        if not valid:
            raise ValueError(f'Invalid training data, doesnt match config.\n'
                             f'Network Input: {repr(self.input_count)}; Network Output: {repr(self.output_count)}\n'
                             f'Points: {repr(training_points)}\nValues: {repr(training_values)}')

    def train(self, training_points: np.ndarray, training_values: np.ndarray,
              status_callback: Optional[Callable[['NeuralNetwork', int], None]] = None, insert_identity_column: bool = True) -> None:
        self._validate_training_data(training_points, training_values)

        if insert_identity_column:
            training_points = NeuralNetwork.with_identity_dimension(training_points)

        while not self.has_training_ended():
            selected_training_point: int = self.do_training_iteration(training_points, training_values)
            if status_callback:
                status_callback(self, selected_training_point)

    def do_training_iteration(self, training_points: np.ndarray, training_values: np.ndarray) -> int:
        # Every (soft_reset_threshold * len(training_points)) iterations, reset weight
        if self.iters_since_soft_reset > self.soft_reset_threshold * len(training_points):
            self.soft_training_reset()

        # Select training point for iteration
        point: int = np.random.randint(0, len(training_points))

        # Calculate activation for point
        activation: Union[float, np.ndarray] = self.predict(training_points[point], training=True)

        # Update direction to use on weight update
        self._update_delta_direction(training_values[point], activation)

        # If appropriate, do linear search to calculate learning rate
        if self.linear_search_l_rate:
            self._recalculate_l_rate_with_linear_search(training_points, training_values)

        # Update weight using current l_rate and delta direction
        self._update_training_weight(training_points[point])

        # Calculate error
        current_error: float = self.calculate_error(training_points, training_values, training=True)

        # Update how many iterations the error has been the same
        if math.isclose(self.last_training_error, current_error, abs_tol=self.training_error_tolerance):
            self.stale_error_count += 1
        else:
            self.stale_error_count = 0

        # Update error and persist training weight if error improved
        has_error_improved: bool = current_error < self.error
        if has_error_improved:
            self.error = current_error
            self._persist_training_weight()

        # If appropriate, recalculate l_rate using variable l_rate strategy
        if not self.linear_search_l_rate:
            self._recalculate_l_rate_with_constant_factor(has_error_improved)

        # Update last training error (for logging and stale error calculation)
        self.last_training_error = current_error

        # Increment iteration count
        self.training_iteration += 1
        self.iters_since_soft_reset += 1

        return point

    @abstractmethod
    def soft_training_reset(self) -> None:
        self.iters_since_soft_reset = 0

    @abstractmethod
    def predict(self, point: np.ndarray, training: bool = False, insert_identity_column: bool = False) -> Union[float, np.ndarray]:
        pass

    @abstractmethod
    def _update_delta_direction(self, training_value: Union[np.ndarray, float], prediction: Union[float, np.ndarray]) -> None:
        pass

    @abstractmethod
    def _update_training_weight(self, training_point: np.ndarray) -> None:
        pass

    @abstractmethod
    def calculate_error(self, training_points: np.ndarray, training_values: np.ndarray, training: bool = False) -> float:
        pass

    @abstractmethod
    def _persist_training_weight(self) -> None:
        pass

    def predict_points(self, points: np.ndarray, training: bool = False, insert_identity_column: bool = False) -> np.ndarray:
        if insert_identity_column:
            points = NeuralNetwork.with_identity_dimension(points)
        return np.apply_along_axis(self.predict, 1, points, training)

    # Retorna los puntos que fueron predichos incorrectamente
    def validate_points(self, points: np.ndarray, values: np.ndarray, error_tolerance: float = DEFAULT_ERROR_TOLERANCE,
                        insert_identity_column: bool = True) -> np.ndarray:
        if insert_identity_column:
            points = NeuralNetwork.with_identity_dimension(points)

        return np.array([point for idx, point in enumerate(points) if not math.isclose(self.predict(point), values[idx], abs_tol=error_tolerance)])

    def _recalculate_l_rate_with_constant_factor(self, has_error_improved: bool) -> None:
        if has_error_improved:
            self.error_positive_trend += 1
            self.error_negative_trend = 0
            if self.error_positive_trend >= self.error_positive_trend_threshold:
                self.l_rate += self.l_rate_up_scaling_factor

        else:
            self.error_negative_trend += 1
            self.error_positive_trend = 0
            if self.error_negative_trend >= self.error_negative_trend_threshold:
                self.l_rate -= self.l_rate_down_scaling_factor * self.l_rate

    def _recalculate_l_rate_with_linear_search(self, training_points: np.ndarray, training_values: np.ndarray) -> None:
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize_scalar.html#scipy.optimize.minimize_scalar
        result: OptimizeResult = minimize_scalar(
            self._test_l_rate,
            bounds=(0, 1),
            args=(training_points, training_values),
            method='bounded',
            options={
                'xatol': self.linear_search_l_rate_error_tolerance,
                'maxiter': self.linear_search_l_rate_max_iterations,
                'disp': 1,  # Imprime cuando no converge. Lo dejo en uno por ahora para ver que onda TODO(tobi): Sacarlo
            }
        )

        if not result.success:
            print(result.message, f'Iterations: {result.nit}')
            raise ValueError('Could not optimize l_rate')

        self.l_rate = result.x

    @abstractmethod
    def _test_l_rate(self, l_rate: float, training_points: np.ndarray, training_values: np.ndarray) -> float:
        pass


    def get_accuracy(self, validation_points: np.ndarray, validation_values: np.ndarray, class_count: int, classify: Callable[[float], float], insert_identity_column: bool = False) -> float:
        classified_points: np.ndarray = np.vectorize(classify)(self.predict_points(validation_points, insert_identity_column=insert_identity_column)).flatten()
        classified_values: np.ndarray = np.vectorize(classify)(validation_values).flatten()
        confusion_matrix: np.ndarray = np.zeros((class_count, class_count))
        for prediction, value in zip(classified_points, classified_values):
            confusion_matrix[value][prediction] += 1

        return confusion_matrix.trace()/np.sum(confusion_matrix)

    def get_precision(self, validation_points: np.ndarray, validation_values: np.ndarray, class_count: int, classify: Callable[[float], float], insert_identity_column: bool = False) -> np.ndarray:
        classified_points: np.ndarray = np.vectorize(classify)(self.predict_points(validation_points, insert_identity_column=insert_identity_column)).flatten()
        classified_values: np.ndarray = np.vectorize(classify)(validation_values).flatten()
        confusion_matrix: np.ndarray = np.zeros((class_count, class_count))
        for prediction, value in zip(classified_points, classified_values):
            confusion_matrix[value][prediction] += 1

        return np.fromiter((confusion_matrix[i][i]/np.sum(confusion_matrix[:, i]) if confusion_matrix[i][i] != 0 else 0 for i in range(class_count)), float)

    def get_recall(self, validation_points: np.ndarray, validation_values: np.ndarray, class_count: int, classify: Callable[[float], float], insert_identity_column: bool = False) -> np.ndarray:
        classified_points: np.ndarray = np.vectorize(classify)(self.predict_points(validation_points, insert_identity_column=insert_identity_column)).flatten()
        classified_values: np.ndarray = np.vectorize(classify)(validation_values).flatten()
        confusion_matrix: np.ndarray = np.zeros((class_count, class_count))
        for prediction, value in zip(classified_points, classified_values):
            confusion_matrix[value][prediction] += 1

        return np.fromiter((confusion_matrix[i][i]/np.sum(confusion_matrix[i]) if confusion_matrix[i][i] != 0 else 0 for i in range(class_count)), float)

    def get_f1_score(self, validation_points: np.ndarray, validation_values: np.ndarray, class_count: int,
                   classify: Callable[[float], float], insert_identity_column: bool = False) -> np.ndarray:
        precision: np.ndarray = self.get_precision(validation_points, validation_values, class_count, classify, insert_identity_column)
        recall: np.ndarray = self.get_recall(validation_points, validation_values, class_count, classify, insert_identity_column)

        return np.fromiter((2 * precision[i] * recall[i] / (precision[i] + recall[i]) if (precision[i] + recall[i]) != 0 else 0 for i in range(class_count)), float)


# Clase interna de las NeuralNetworks
# Todos los puntos que se reciben en los métodos deben venir con la columna identidad
class _Perceptron:

    def __init__(self, input_count: int, activation_fn: ActivationFunction, momentum_factor: float):
        self.input_count = input_count
        self.activation_fn: ActivationFunction = activation_fn
        self.momentum_factor: float = momentum_factor
        self.w: np.ndarray
        self.training_w: np.ndarray

        self.training_weights_reset()  # Inicializar training_w
        self.persist_weights()         # Inicializar w

        # Caches
        self.last_excitement: float = 0
        self.last_activation: float = 0
        self.delta: float = 0
        self._last_delta_w: np.ndarray = np.zeros(input_count + 1)  # Para implementar momentum

    def predict(self, point: np.ndarray, training: bool = False) -> float:
        weights: np.ndarray = (self.training_w if training else self.w)

        self.last_excitement = sum(point * weights)
        self.last_activation = self.activation_fn(self.last_excitement)

        return self.last_activation

    def persist_weights(self) -> None:
        self.w = self.training_w

    def update_training_weights(self, l_rate: float, point: np.ndarray) -> None:
        new_delta_w: np.ndarray = l_rate * self.delta * point + self.momentum_factor * self._last_delta_w
        self.training_w += new_delta_w
        self._last_delta_w = new_delta_w

    def test_l_rate(self, point: np.ndarray, l_rate: float) -> float:
        test_weight: np.ndarray = self.training_w + l_rate * self.delta * point + self.momentum_factor * self._last_delta_w
        return self.activation_fn(sum(point * test_weight))

    def training_weights_reset(self) -> None:
        self.training_w = np.random.uniform(-1, 1, self.input_count + 1)


class SinglePerceptronNeuralNetwork(NeuralNetwork, ABC):

    def __init__(self, base_config: NeuralNetworkBaseConfiguration) -> None:
        base_config.output_count = 1
        super().__init__(base_config)
        self._perceptron = _Perceptron(self.input_count, self.activation_fn, self.momentum_factor)

    def train(self, training_points: np.ndarray, training_values: np.ndarray,
              status_callback: Optional[Callable[['NeuralNetwork', int], None]] = None, insert_identity_column: bool = True) -> None:
        training_values = np.squeeze(training_values)
        super().train(training_points, training_values, status_callback, insert_identity_column)

    def soft_training_reset(self) -> None:
        super().soft_training_reset()
        self._perceptron.training_weights_reset()

    def predict(self, point: np.ndarray, training: bool = False, insert_identity_column: bool = False) -> Union[float, np.ndarray]:
        if insert_identity_column:
            point = NeuralNetwork.with_identity_dimension(point, 0)
        return self._perceptron.predict(point, training)

    def _update_delta_direction(self, training_value: Union[np.ndarray, float], activation: Union[float, np.ndarray]) -> None:
        if not isinstance(training_value, float) or not isinstance(activation, float):
            raise TypeError('training value and activation mast be a float')
        self._perceptron.delta = self._calculate_delta(training_value, activation)

    @abstractmethod
    def _calculate_delta(self, training_value: float, activation: float) -> float:
        pass

    def _update_training_weight(self, training_point: np.ndarray) -> None:
        self._perceptron.update_training_weights(self.l_rate, training_point)

    def calculate_error(self, points: np.ndarray, values: np.ndarray, training: bool = False) -> float:
        return self.error_function.calculate_error(values, self.predict_points(points, training))

    def _persist_training_weight(self) -> None:
        self._perceptron.persist_weights()

    def _test_l_rate(self, l_rate: float, training_points: np.ndarray, training_values: np.ndarray) -> float:
        tests: np.ndarray = np.apply_along_axis(self._perceptron.test_l_rate, 1, training_points, l_rate)
        return self.error_function.calculate_error(training_values, tests)


class SimpleSinglePerceptronNeuralNetwork(SinglePerceptronNeuralNetwork):

    def __init__(self, base_config: NeuralNetworkBaseConfiguration) -> None:
        base_config.activation_fn = np.sign
        base_config.error_function = NeuralNetworkErrorFunction.ABSOLUTE
        super().__init__(base_config)

    def _calculate_delta(self, training_value: float, activation: float) -> float:
        return self.error_function.calculate_error_factor(training_value, activation)


class LinearSinglePerceptronNeuralNetwork(SinglePerceptronNeuralNetwork):

    def __init__(self, base_config: NeuralNetworkBaseConfiguration) -> None:
        base_config.activation_fn = lambda x: x
        base_config.error_function = NeuralNetworkErrorFunction.QUADRATIC
        super().__init__(base_config)

    def _calculate_delta(self, training_value: float, activation: float) -> float:
        return self.error_function.calculate_error_factor(training_value, activation)


class NonLinearSinglePerceptronNeuralNetwork(SinglePerceptronNeuralNetwork):

    def __init__(self,
                 base_config: NeuralNetworkBaseConfiguration,
                 activation_derivative: ActivationFunction,
                 error_function: NeuralNetworkErrorFunction) -> None:
        base_config.error_function = error_function
        super().__init__(base_config)
        self.activation_derivative: ActivationFunction = activation_derivative
        if self.error_function == NeuralNetworkErrorFunction.ABSOLUTE:
            raise ValueError(f'Invalid error function {self.error_function.name} for {repr(type(self))}')

    def _calculate_delta(self, training_value: float, activation: float) -> float:
        return (
            self.error_function.calculate_error_factor(training_value, activation) *
            self.activation_derivative(self._perceptron.last_excitement)
        )


class _PerceptronLayer:

    def __init__(self, size: int, perceptron_factory: Callable[[], _Perceptron], activation_derivative: ActivationFunction) -> None:
        self.size = size
        self.perceptrons: List[_Perceptron] = [perceptron_factory() for _ in range(self.size)]
        self.activation_derivative: ActivationFunction = activation_derivative
        self.activation_cache: np.ndarray = np.zeros(self.size + 1)

    def predict(self, point: np.ndarray, training: bool = False) -> np.ndarray:
        self.activation_cache = np.fromiter(
            itertools.chain(
                # Concatenar el primer valor identidad (1) con las predicciones de los perceptrons
                range(1, 2),
                (perceptron.predict(point, training) for perceptron in self.perceptrons)
            ),
            float
        )
        return self.activation_cache

    def update_perceptrons_delta(self, delta_multiplier: np.ndarray) -> None:
        for i, perceptron in enumerate(self.perceptrons):
            perceptron.delta = self.activation_derivative(perceptron.last_excitement) * delta_multiplier[i]

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

    def update_w(self, l_rate: float, previous_activations: np.ndarray):
        for perceptron in self.perceptrons:
            perceptron.update_training_weights(l_rate, previous_activations)

    def test_l_rate(self, point: np.ndarray, l_rate: float) -> np.ndarray:
        return np.fromiter(
            itertools.chain(
                # Concatenar el primer valor identidad (1) con los tests de los perceptrons
                range(1, 2),
                (perceptron.test_l_rate(point, l_rate) for perceptron in self.perceptrons)
            ),
            float
        )

    def persist_training_weights(self) -> None:
        for perceptron in self.perceptrons:
            perceptron.persist_weights()

    def training_weights_reset(self) -> None:
        for perceptron in self.perceptrons:
            perceptron.training_weights_reset()

class MultilayeredNeuralNetwork(NeuralNetwork):

    def _validate_layer_config(self, layer_sizes: List[int]) -> None:
        valid: bool = (
            len(layer_sizes) > 0 and
            functools.reduce(lambda valid, layer_size: valid and layer_size > 0, layer_sizes, True)  # Todos valores positivos
        )
        if not valid:
            raise ValueError(f'Invalid Layer Sizes in {repr(type(self))}:\n{repr(layer_sizes)}')

    def __init__(self, base_config: NeuralNetworkBaseConfiguration,
                 activation_derivative: ActivationFunction,
                 error_function: NeuralNetworkErrorFunction,
                 layer_sizes: List[int]) -> None:

        base_config.error_function = error_function
        self._validate_layer_config(layer_sizes)
        base_config.output_count = layer_sizes[-1]
        super().__init__(base_config)
        if self.error_function == NeuralNetworkErrorFunction.ABSOLUTE:
            raise ValueError(f'Invalid error function {self.error_function.name} for {repr(type(self))}')

        perceptron_factory: Callable[[int], _Perceptron] = \
            lambda input_count: _Perceptron(
                input_count, self.activation_fn, self.momentum_factor
            )

        # layer_sizes[i + 1] = layer_size => layer_sizes[i] = perceptrons input size
        # Los perceptrones de la capa actual reciben una cantidad de inputs equivalente al tamaño de la layer anterior
        layer_sizes = [self.input_count] + layer_sizes  # La primera capa recibe input_count inputs
        self._layers: List[_PerceptronLayer] = [
            _PerceptronLayer(layer_sizes[i + 1], lambda: perceptron_factory(layer_sizes[i]), activation_derivative)
            for i in range(len(layer_sizes) - 1)
        ]

    def soft_training_reset(self) -> None:
        super().soft_training_reset()
        for m in range(len(self._layers)):
            self._layers[m].training_weights_reset()

    def predict(self, point: np.ndarray, training: bool = False, insert_identity_column: bool = False) -> Union[float, np.ndarray]:
        if insert_identity_column:
            point = NeuralNetwork.with_identity_dimension(point, 0)

        last_prediction: np.ndarray = point

        # Conseguir el V de la capa de salida, propagando los puntos de entrada por todas las capas
        for m in range(len(self._layers)):
            last_prediction = self._layers[m].predict(last_prediction, training)

        return last_prediction[1:]  # Viene con la identity column agregada de mas

    def _update_delta_direction(self, training_value: Union[np.ndarray, float], activation: Union[float, np.ndarray]) -> None:
        if not isinstance(training_value, np.ndarray) or not isinstance(activation, np.ndarray):
            raise TypeError('training value and activation must be an ndarray')
        self._update_deltas(training_value, activation)

    def _update_training_weight(self, training_point: np.ndarray) -> None:
        previous_activation: np.ndarray = training_point

        for m in range(len(self._layers)):
            self._layers[m].update_w(self.l_rate, previous_activation)
            previous_activation = self._layers[m].activation_cache

    def calculate_error(self, points: np.ndarray, values: np.ndarray, training: bool = False) -> float:
        return self.error_function.calculate_error(values.flatten(), self.predict_points(points, training).flatten())

    def _persist_training_weight(self) -> None:
        for m in range(len(self._layers)):
            self._layers[m].persist_training_weights()

    def _update_deltas(self, training_value: np.ndarray, prediction: np.ndarray) -> None:
        # TODO(tobi): Se puede hacer mejor?
        delta_multiplier: np.ndarray = np.fromiter(
            (self.error_function.calculate_error_factor(training_value[i], prediction[i]) for i in range(len(training_value))),
            float
        )

        for m in range(len(self._layers) - 1, -1, -1):
            self._layers[m].update_perceptrons_delta(delta_multiplier)
            delta_multiplier = self._layers[m].calculate_previous_layer_delta_multiplier()

    def _test_l_rate_one_point(self, point: np.ndarray, l_rate: float) -> np.ndarray:
        last_test: np.ndarray = point

        # Conseguir el V de la capa de salida, propagando los puntos de entrada por todas las capas
        for m in range(len(self._layers)):
            last_test = self._layers[m].test_l_rate(last_test, l_rate)

        return last_test[1:]  # Viene con la identity column

    def _test_l_rate(self, l_rate: float, training_points: np.ndarray, training_values: np.ndarray) -> float:
        tests: np.ndarray = np.apply_along_axis(self._test_l_rate_one_point, 1, training_points, l_rate)
        return self.error_function.calculate_error(training_values.flatten(), tests.flatten())
