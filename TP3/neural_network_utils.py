from typing import Dict, Callable, Tuple

import numpy as np
from schema import And, Schema, Or, Optional

from config import Param, Config
from neural_network import MultilayeredNeuralNetwork, SimpleSinglePerceptronNeuralNetwork, \
    LinearSinglePerceptronNeuralNetwork, \
    NonLinearSinglePerceptronNeuralNetwork, ActivationFunction, NeuralNetwork, NeuralNetworkBaseConfiguration, \
    NeuralNetworkErrorFunction

NeuralNetworkFactory = Callable[[NeuralNetworkBaseConfiguration, Param], NeuralNetwork]
SigmoidFunction = Callable[[float, float], float]
SigmoidDerivativeFunction = SigmoidFunction
MetricCalculator = Callable[[NeuralNetwork, np.ndarray, np.ndarray], float]

def _validate_base_network_params(perceptron_params: Param) -> Param:
    return Config.validate_param(perceptron_params, Schema({
        'type': And(str, Or(*tuple(_neural_network_factory_map.keys()))),
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


def get_neural_network(base_network_params: Param, input_count: int) -> NeuralNetwork:
    base_network_params = _validate_base_network_params(base_network_params)

    base_network_config: NeuralNetworkBaseConfiguration = _build_base_network_config(base_network_params, input_count)

    factory: NeuralNetworkFactory = _neural_network_factory_map[base_network_params['type']]

    return factory(base_network_config, base_network_params['network_params'])


def _get_simple_perceptron(base_config: NeuralNetworkBaseConfiguration, params: Param) -> NeuralNetwork:
    return SimpleSinglePerceptronNeuralNetwork(base_config)


def _get_linear_perceptron(base_config: NeuralNetworkBaseConfiguration, params: Param) -> NeuralNetwork:
    return LinearSinglePerceptronNeuralNetwork(base_config)


def _validate_non_linear_perceptron_params(params: Param) -> Param:
    return Config.validate_param(params, Schema({
        'activation_function': And(str, Or(*tuple(_sigmoid_activation_function_map.keys()))),
        'activation_slope_factor': And(Or(float, int), lambda b: b > 0),
        'error_function': And(str, Or(*tuple(_neural_network_error_function_map.keys())))
    }, ignore_extra_keys=True))


def _get_non_linear_perceptron(base_config: NeuralNetworkBaseConfiguration, params: Param) -> NeuralNetwork:
    params = _validate_non_linear_perceptron_params(params)
    activation_function, activation_derivative = _sigmoid_activation_function_map[params['activation_function']]

    error_function: NeuralNetworkErrorFunction = _neural_network_error_function_map[params['error_function']]
    base_config.activation_fn = lambda x: activation_function(x, params['activation_slope_factor'])
    real_activation_derivative: ActivationFunction = lambda x: activation_derivative(x, params['activation_slope_factor'])

    return NonLinearSinglePerceptronNeuralNetwork(base_config, real_activation_derivative, error_function)


def _validate_multi_layered_perceptron_params(params: Param) -> Param:
    return Config.validate_param(params, Schema({
        'activation_function': And(str, Or(*tuple(_sigmoid_activation_function_map.keys()))),
        'activation_slope_factor': And(Or(float, int), lambda b: b > 0),
        'error_function': And(str, Or(*tuple(_neural_network_error_function_map.keys()))),
        'layer_sizes': And(list)  # Se puede validar que sean int > 0 aca con una funcion. Ya se hace en el proceso de validacion interno de la libreria
    }, ignore_extra_keys=True))


def _get_multi_layered_perceptron(base_config: NeuralNetworkBaseConfiguration, params: Param) -> NeuralNetwork:
    params = _validate_multi_layered_perceptron_params(params)
    activation_function, activation_derivative = _sigmoid_activation_function_map[params['activation_function']]

    error_function: NeuralNetworkErrorFunction = _neural_network_error_function_map[params['error_function']]
    base_config.activation_fn = lambda x: activation_function(x, params['activation_slope_factor'])
    real_activation_derivative: ActivationFunction = lambda x: activation_derivative(x, params['activation_slope_factor'])

    return MultilayeredNeuralNetwork(base_config, real_activation_derivative, error_function, params['layer_sizes'])


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

_neural_network_factory_map: Dict[str, NeuralNetworkFactory] = {
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


def _accuracy(nn: NeuralNetwork, points: np.ndarray, values: np.ndarray) -> float:
    return nn.get_accuracy(points, values, 2, lambda x: 1 if x >= 0 else 0, insert_identity_column=True)


def _positive_precision(nn: NeuralNetwork, points: np.ndarray, values: np.ndarray) -> float:
    return nn.get_precision(points, values, 2, lambda x: 1 if x >= 0 else 0, insert_identity_column=True)[0]


def _negative_precision(nn: NeuralNetwork, points: np.ndarray, values: np.ndarray) -> float:
    return nn.get_precision(points, values, 2, lambda x: 1 if x >= 0 else 0, insert_identity_column=True)[1]


def _positive_recall(nn: NeuralNetwork, points: np.ndarray, values: np.ndarray) -> float:
    return nn.get_recall(points, values, 2, lambda x: 1 if x >= 0 else 0, insert_identity_column=True)[0]


def _negative_recall(nn: NeuralNetwork, points: np.ndarray, values: np.ndarray) -> float:
    return nn.get_recall(points, values, 2, lambda x: 1 if x >= 0 else 0, insert_identity_column=True)[1]


def _positive_f1score(nn: NeuralNetwork, points: np.ndarray, values: np.ndarray) -> float:
    return nn.get_f1_score(points, values, 2, lambda x: 1 if x >= 0 else 0, insert_identity_column=True)[0]


def _negative_f1score(nn: NeuralNetwork, points: np.ndarray, values: np.ndarray) -> float:
    return nn.get_f1_score(points, values, 2, lambda x: 1 if x >= 0 else 0, insert_identity_column=True)[1]


_neural_network_metrics: Dict[str, MetricCalculator] = {
    'error': _error,
    'accuracy': _accuracy,
    'positive_precision': _positive_precision,
    'negative_precision': _negative_precision,
    'positive_recall': _positive_recall,
    'negative_recall': _negative_recall,
    'positive_f1score': _positive_f1score,
    'negative_f1score': _negative_f1score,
}