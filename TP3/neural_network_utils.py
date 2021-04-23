import bisect
from typing import Dict, Callable, Tuple, List, TypeVar

import numpy as np
import typing
from schema import And, Schema, Or, Optional
import attr

from config import Param, Config
from neural_network import MultilayeredNeuralNetwork, SimpleSinglePerceptronNeuralNetwork, \
    LinearSinglePerceptronNeuralNetwork, \
    NonLinearSinglePerceptronNeuralNetwork, ActivationFunction, NeuralNetwork, NeuralNetworkBaseConfiguration, \
    NeuralNetworkErrorFunction

_NeuralNetworkFactoryBuilder = Callable[[NeuralNetworkBaseConfiguration, Param], Callable[[], NeuralNetwork]]

NeuralNetworkFactory = Callable[[], NeuralNetwork]
SigmoidFunction = Callable[[float, float], float]
SigmoidDerivativeFunction = SigmoidFunction
MetricCalculator = Callable[[NeuralNetwork, np.ndarray, np.ndarray], float]

# Generic Internal Variable
_T = TypeVar('_T')


def _assert_not_none(obj: typing.Optional[_T]) -> _T:
    if obj is None:
        raise TypeError()
    return obj


@attr.s(auto_attribs=True)
class CrossValidationResult:
    best_neural_network: NeuralNetwork
    best_test_points: np.ndarray
    best_test_values: np.ndarray
    metrics_mean: float
    metrics_std: float


def cross_validation(neural_network_factory: NeuralNetworkFactory,
                     training_points: np.ndarray, training_values: np.ndarray,
                     get_metric: MetricCalculator,
                     test_points_count: int,
                     iteration_count: int) -> CrossValidationResult:
    if (
        test_points_count > len(training_values)//2 or
        test_points_count <= 0 or
        len(training_values) != len(training_points) or
        iteration_count <= 0
    ):
        raise ValueError('Invalid cross validation parameters')

    gt_points: np.ndarray
    gt_values: np.ndarray
    gv_points: np.ndarray
    gv_values: np.ndarray
    current_metric: float
    neural_network: NeuralNetwork
    best_metric: typing.Optional[float] = None
    best_indexes: np.ndarray = np.zeros((1, 1))
    best_neural_network: typing.Optional[NeuralNetwork] = None
    all_metrics: List[float] = []

    for _ in range(iteration_count):
        possible_values: np.ndarray = np.arange(len(training_points))

        while len(possible_values) // test_points_count > 0:
            neural_network = neural_network_factory()

            indexes = np.random.choice(possible_values, size=test_points_count, replace=False)
            possible_values = possible_values[~np.isin(possible_values, indexes)]
            gt_points = np.delete(training_points, indexes, axis=0)
            gt_values = np.delete(training_values, indexes, axis=0)
            gv_points = np.take(training_points, indexes, axis=0)
            gv_values = np.take(training_values, indexes, axis=0)

            neural_network.train(gt_points, gt_values)
            current_metric = get_metric(neural_network, gv_points, gv_values)
            all_metrics.append(current_metric)
            if best_metric is None or best_metric < current_metric:
                best_metric = current_metric
                best_indexes = indexes
                best_neural_network = neural_network

    all_metrics_np: np.ndarray = np.array(all_metrics)
    return CrossValidationResult(
        _assert_not_none(best_neural_network),
        np.take(training_points, best_indexes, axis=0),
        np.take(training_values, best_indexes, axis=0),
        all_metrics_np.mean(),
        all_metrics_np.std(),
    )


def _validate_base_network_params(perceptron_params: Param) -> Param:
    return Config.validate_param(perceptron_params, Schema({
        'type': And(str, Or(*tuple(_neural_network_factory_builder_map.keys()))),
        Optional('max_training_iterations', default=None): And(int, lambda i: i > 0),
        Optional('weight_reset_threshold', default=None): And(int, lambda i: i > 0),
        Optional('max_stale_error_iterations', default=None): And(int, lambda i: i > 0),
        Optional('error_goal', default=None): And(Or(float, int), lambda i: i > 0),
        Optional('error_tolerance', default=None): And(Or(float, int), lambda i: i > 0),
        Optional('momentum_factor', default=None): And(Or(float, int), lambda i: i >= 0),
        Optional('learning_rate_strategy', default='variable'): Or('fixed', 'variable', 'linear_search'),
        Optional('base_learning_rate', default=None): And(Or(float, int), lambda i: i > 0),
        Optional('variable_learning_rate_params', default=dict): {
            Optional('up_scaling_factor', default=None): And(Or(float, int), lambda i: i > 0),
            Optional('down_scaling_factor', default=None): And(Or(float, int), lambda i: i > 0),
            Optional('positive_trend_threshold', default=None): And(int, lambda i: i > 0),
            Optional('negative_trend_threshold', default=None): And(int, lambda i: i > 0),
        },
        Optional('learning_rate_linear_search_params', default=dict): {
            Optional('max_iterations', default=None): And(int, lambda i: i > 0),
            Optional('error_tolerance', default=None): And(Or(int, float), lambda i: i > 0),
            Optional('max_value', default=None): And(int, lambda i: i > 0),
        },
        Optional('network_params', default=dict): dict,
    }, ignore_extra_keys=True))


def _build_base_network_config(network_params: Param, input_count: int) -> NeuralNetworkBaseConfiguration:
    ret: NeuralNetworkBaseConfiguration = NeuralNetworkBaseConfiguration(input_count=input_count)
    if network_params['max_training_iterations'] is not None: ret.max_training_iterations = network_params['max_training_iterations']
    if network_params['weight_reset_threshold'] is not None: ret.soft_reset_threshold = network_params['weight_reset_threshold']
    if network_params['max_stale_error_iterations'] is not None: ret.max_stale_error_iterations = network_params['max_stale_error_iterations']
    if network_params['error_goal'] is not None: ret.training_error_goal = network_params['error_goal']
    if network_params['error_tolerance'] is not None: ret.training_error_tolerance = network_params['error_tolerance']
    if network_params['momentum_factor'] is not None: ret.momentum_factor = network_params['momentum_factor']
    if network_params['base_learning_rate'] is not None: ret.base_l_rate = network_params['base_learning_rate']
    ret.linear_search_l_rate = network_params['learning_rate_strategy'] == 'linear_search'
    # Learning Rate Linear Search Params
    if ret.linear_search_l_rate:
        l_rate_linear_search_params: Param = network_params['learning_rate_linear_search_params']
        if l_rate_linear_search_params['max_iterations'] is not None: ret.linear_search_l_rate_max_iterations = l_rate_linear_search_params['max_iterations']
        if l_rate_linear_search_params['max_value'] is not None: ret.linear_search_max_value = l_rate_linear_search_params['max_value']
        if l_rate_linear_search_params['error_tolerance'] is not None: ret.linear_search_l_rate_error_tolerance = l_rate_linear_search_params['error_tolerance']
        # Variable Learning Rate Params
    if network_params['learning_rate_strategy'] == 'variable':
        variable_l_rate_params: Param = network_params['variable_learning_rate_params']
        if variable_l_rate_params['up_scaling_factor'] is not None: ret.l_rate_up_scaling_factor = variable_l_rate_params['up_scaling_factor']
        if variable_l_rate_params['down_scaling_factor'] is not None: ret.l_rate_down_scaling_factor = variable_l_rate_params['down_scaling_factor']
        if variable_l_rate_params['positive_trend_threshold'] is not None: ret.error_positive_trend_threshold = variable_l_rate_params['positive_trend_threshold']
        if variable_l_rate_params['negative_trend_threshold'] is not None: ret.error_negative_trend_threshold = variable_l_rate_params['negative_trend_threshold']

    return ret


def get_neural_network_factory(base_network_params: Param, input_count: int) -> NeuralNetworkFactory:
    base_network_params = _validate_base_network_params(base_network_params)

    base_network_config: NeuralNetworkBaseConfiguration = _build_base_network_config(base_network_params, input_count)

    factory_builder: _NeuralNetworkFactoryBuilder = _neural_network_factory_builder_map[base_network_params['type']]

    return factory_builder(base_network_config, base_network_params['network_params'])


def get_neural_network(base_network_params: Param, input_count: int) -> NeuralNetwork:
    neural_network_factory: NeuralNetworkFactory = get_neural_network_factory(base_network_params, input_count)
    return neural_network_factory()


def _get_simple_perceptron(base_config: NeuralNetworkBaseConfiguration, params: Param) -> Callable[[], NeuralNetwork]:
    return lambda: SimpleSinglePerceptronNeuralNetwork(base_config)


def _get_linear_perceptron(base_config: NeuralNetworkBaseConfiguration, params: Param) -> Callable[[], NeuralNetwork]:
    return lambda: LinearSinglePerceptronNeuralNetwork(base_config)


def _validate_non_linear_perceptron_params(params: Param) -> Param:
    return Config.validate_param(params, Schema({
        'activation_function': And(str, Or(*tuple(_sigmoid_activation_function_map.keys()))),
        'activation_slope_factor': And(Or(float, int), lambda b: b > 0),
        'error_function': And(str, Or(*tuple(_neural_network_error_function_map.keys())))
    }, ignore_extra_keys=True))


def _get_non_linear_perceptron(base_config: NeuralNetworkBaseConfiguration, params: Param) -> Callable[[], NeuralNetwork]:
    params = _validate_non_linear_perceptron_params(params)
    activation_function, activation_derivative = _sigmoid_activation_function_map[params['activation_function']]

    error_function: NeuralNetworkErrorFunction = _neural_network_error_function_map[params['error_function']]
    base_config.activation_fn = lambda x: activation_function(x, params['activation_slope_factor'])
    real_activation_derivative: ActivationFunction = lambda x: activation_derivative(x, params['activation_slope_factor'])

    return lambda: NonLinearSinglePerceptronNeuralNetwork(base_config, real_activation_derivative, error_function)


def _validate_multi_layered_perceptron_params(params: Param) -> Param:
    return Config.validate_param(params, Schema({
        'activation_function': And(str, Or(*tuple(_sigmoid_activation_function_map.keys()))),
        'activation_slope_factor': And(Or(float, int), lambda b: b > 0),
        'error_function': And(str, Or(*tuple(_neural_network_error_function_map.keys()))),
        'layer_sizes': And(list)  # Se puede validar que sean int > 0 aca con una funcion. Ya se hace en el proceso de validacion interno de la libreria
    }, ignore_extra_keys=True))


def _get_multi_layered_perceptron(base_config: NeuralNetworkBaseConfiguration, params: Param) -> Callable[[], NeuralNetwork]:
    params = _validate_multi_layered_perceptron_params(params)
    activation_function, activation_derivative = _sigmoid_activation_function_map[params['activation_function']]

    error_function: NeuralNetworkErrorFunction = _neural_network_error_function_map[params['error_function']]
    base_config.activation_fn = lambda x: activation_function(x, params['activation_slope_factor'])
    real_activation_derivative: ActivationFunction = lambda x: activation_derivative(x, params['activation_slope_factor'])

    return lambda: MultilayeredNeuralNetwork(base_config, real_activation_derivative, error_function, params['layer_sizes'])


# Sigmoid Activation Functions And Derivatives

# Hyperbolic Tangent
def tanh(x: float, b: float) -> float:
    return np.tanh(b * x)


def tanh_derivative(x: float, b: float) -> float:
    return b * (1 - tanh(x, b) ** 2)


# Logistic Function
def logistic(x: float, b: float) -> float:
    return 1 / (1 + np.exp(-2 * b * x))


def logistic_derivative(x: float, b: float) -> float:
    return 2 * b * logistic(x, b) * (1 - logistic(x, b))


# Name to Implementation maps

_neural_network_factory_builder_map: Dict[str, _NeuralNetworkFactoryBuilder] = {
    'simple': _get_simple_perceptron,
    'linear': _get_linear_perceptron,
    'non_linear': _get_non_linear_perceptron,
    'multi_layered': _get_multi_layered_perceptron,
}

_neural_network_error_function_map: Dict[str, NeuralNetworkErrorFunction] = {
    'quadratic': NeuralNetworkErrorFunction.QUADRATIC,
    'logarithmic': NeuralNetworkErrorFunction.LOGARITHMIC,
}

_sigmoid_activation_function_map: Dict[str, Tuple[SigmoidFunction, SigmoidDerivativeFunction]] = {
    'tanh': (tanh, tanh_derivative),
    'logistic': (logistic, logistic_derivative),
}


def _error(nn: NeuralNetwork, points: np.ndarray, values: np.ndarray) -> float:
    return nn.calculate_error(points, values, training=False, insert_identity_column=True)


def _accuracy(nn: NeuralNetwork, points: np.ndarray, values: np.ndarray, class_separators: List[float]) -> float:
    return NeuralNetwork.get_accuracy(
        nn.get_confusion_matrix(points, values, len(class_separators), lambda x: bisect.bisect_left(class_separators, x), insert_identity_column=True)
    )

# TODO: Para mi (tobi) va a estar dificil generalizar esta parte a config.
#  Yo primero me centraria en elegir una opcion muy buena para cada ejercicio.
# def _positive_precision(nn: NeuralNetwork, points: np.ndarray, values: np.ndarray) -> float:
#     return nn.get_precision(points, values, 2, lambda x: 1 if x >= 0 else 0, insert_identity_column=True)[0]
#
#
# def _negative_precision(nn: NeuralNetwork, points: np.ndarray, values: np.ndarray) -> float:
#     return nn.get_precision(points, values, 2, lambda x: 1 if x >= 0 else 0, insert_identity_column=True)[1]
#
#
# def _positive_recall(nn: NeuralNetwork, points: np.ndarray, values: np.ndarray) -> float:
#     return nn.get_recall(points, values, 2, lambda x: 1 if x >= 0 else 0, insert_identity_column=True)[0]
#
#
# def _negative_recall(nn: NeuralNetwork, points: np.ndarray, values: np.ndarray) -> float:
#     return nn.get_recall(points, values, 2, lambda x: 1 if x >= 0 else 0, insert_identity_column=True)[1]
#
#
# def _positive_f1score(nn: NeuralNetwork, points: np.ndarray, values: np.ndarray) -> float:
#     return nn.get_f1_score(points, values, 2, lambda x: 1 if x >= 0 else 0, insert_identity_column=True)[0]
#
#
# def _negative_f1score(nn: NeuralNetwork, points: np.ndarray, values: np.ndarray) -> float:
#     return nn.get_f1_score(points, values, 2, lambda x: 1 if x >= 0 else 0, insert_identity_column=True)[1]


_neural_network_metrics: Dict[str, MetricCalculator] = {
    'error': _error,
    'accuracy': _accuracy,
    # 'positive_precision': _positive_precision,
    # 'negative_precision': _negative_precision,
    # 'positive_recall': _positive_recall,
    # 'negative_recall': _negative_recall,
    # 'positive_f1score': _positive_f1score,
    # 'negative_f1score': _negative_f1score,
}