from math import exp, log
from typing import Dict, Callable, Tuple

import numpy as np
import pandas as pd
from schema import And, Schema, Or, Optional

from TP3.config import Param, Config
from TP3.neural_network_lib.neural_network_utils import NeuralNetworkFactory

from TP3.neural_network_lib.neural_network import MultilayeredNeuralNetwork, SimpleSinglePerceptronNeuralNetwork, \
    LinearSinglePerceptronNeuralNetwork, \
    NonLinearSinglePerceptronNeuralNetwork, ActivationFunction, NeuralNetwork, NeuralNetworkBaseConfiguration, \
    NeuralNetworkErrorFunction

_NeuralNetworkFactoryBuilder = Callable[[NeuralNetworkBaseConfiguration, Param], Callable[[], NeuralNetwork]]

SigmoidFunction = Callable[[float, float], float]
SigmoidDerivativeFunction = SigmoidFunction


def get_training_set(file_name: str, line_count: int, normalize: bool) -> np.ndarray:
    training_set: np.ndarray = pd.read_csv(file_name, delim_whitespace=True, header=None).values
    if normalize:
        training_set = training_set / 100

    if line_count > 1:
        elem_size: int = len(training_set[0]) * line_count
        training_set = np.reshape(training_set, (np.size(training_set) // elem_size, elem_size))

    return training_set


def _validate_base_network_params(perceptron_params: Param) -> Param:
    return Config.validate_param(perceptron_params, Schema({
        'type': And(str, Or(*tuple(_neural_network_factory_builder_map.keys()))),
        Optional('max_training_iterations', default=None): And(int, lambda i: i > 0),
        Optional('weight_reset_threshold', default=None): And(int, lambda i: i > 0),
        Optional('max_stale_error_iterations', default=None): And(int, lambda i: i > 0),
        Optional('error_goal', default=None): And(Or(float, int), lambda i: i >= 0),
        Optional('error_tolerance', default=None): And(Or(float, int), lambda i: i > 0),
        Optional('momentum_factor', default=None): And(Or(float, int), lambda i: i >= 0),
        Optional('learning_rate_strategy', default='fixed'): Or('fixed', 'variable', 'linear_search'),
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


def build_base_network_config(network_params: Param, input_count: int) -> NeuralNetworkBaseConfiguration:
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
    if ret.linear_search_l_rate and network_params['learning_rate_linear_search_params']:
        l_rate_linear_search_params: Param = network_params['learning_rate_linear_search_params']
        if l_rate_linear_search_params['max_iterations'] is not None: ret.linear_search_l_rate_max_iterations = l_rate_linear_search_params['max_iterations']
        if l_rate_linear_search_params['max_value'] is not None: ret.linear_search_max_value = l_rate_linear_search_params['max_value']
        if l_rate_linear_search_params['error_tolerance'] is not None: ret.linear_search_l_rate_error_tolerance = l_rate_linear_search_params['error_tolerance']
        # Variable Learning Rate Params
    if network_params['learning_rate_strategy'] == 'variable' and network_params['variable_learning_rate_params']:
        variable_l_rate_params: Param = network_params['variable_learning_rate_params']
        if variable_l_rate_params['up_scaling_factor'] is not None: ret.l_rate_up_scaling_factor = variable_l_rate_params['up_scaling_factor']
        if variable_l_rate_params['down_scaling_factor'] is not None: ret.l_rate_down_scaling_factor = variable_l_rate_params['down_scaling_factor']
        if variable_l_rate_params['positive_trend_threshold'] is not None: ret.error_positive_trend_threshold = variable_l_rate_params['positive_trend_threshold']
        if variable_l_rate_params['negative_trend_threshold'] is not None: ret.error_negative_trend_threshold = variable_l_rate_params['negative_trend_threshold']

    return ret


def get_neural_network_factory(base_network_params: Param, input_count: int) -> NeuralNetworkFactory:
    base_network_params = _validate_base_network_params(base_network_params)

    base_network_config: NeuralNetworkBaseConfiguration = build_base_network_config(base_network_params, input_count)

    factory_builder: _NeuralNetworkFactoryBuilder = _neural_network_factory_builder_map[base_network_params['type']]

    return factory_builder(base_network_config, base_network_params['network_params'])


def get_neural_network(base_network_params: Param, input_count: int) -> NeuralNetwork:
    neural_network_factory: NeuralNetworkFactory = get_neural_network_factory(base_network_params, input_count)
    return neural_network_factory()


def _get_simple_perceptron(base_config: NeuralNetworkBaseConfiguration, params: Param) -> Callable[[], NeuralNetwork]:
    return lambda: SimpleSinglePerceptronNeuralNetwork(base_config)


def _validate_linear_perceptron_params(params: Param) -> Param:
    return Config.validate_param(params, Schema({
        'error_function': And(str, Or(*tuple(_neural_network_error_function_map.keys())))
    }, ignore_extra_keys=True))


def _get_linear_perceptron(base_config: NeuralNetworkBaseConfiguration, params: Param) -> Callable[[], NeuralNetwork]:
    params = _validate_linear_perceptron_params(params)
    error_function: NeuralNetworkErrorFunction = _neural_network_error_function_map[params['error_function']]

    return lambda: LinearSinglePerceptronNeuralNetwork(base_config, error_function)


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

# Sigmoid
def sigmoid(x: float, b: float) -> float:
    return 1 / (1 + np.exp(-x * b))


def sigmoid_derivative(x: float, b: float) -> float:
    return sigmoid(x, b)*(1 - sigmoid(x, b))

# Sigmoid
def softplus(x: float, b: float) -> float:
    return np.log(1 + np.exp(x * b))


def softplus_derivative(x: float, b: float) -> float:
    return 1 / (1 + np.exp(-x * b))


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
    'absolute': NeuralNetworkErrorFunction.ABSOLUTE,
    'quadratic': NeuralNetworkErrorFunction.QUADRATIC,
    'logarithmic': NeuralNetworkErrorFunction.LOGARITHMIC,
}

_sigmoid_activation_function_map: Dict[str, Tuple[SigmoidFunction, SigmoidDerivativeFunction]] = {
    'tanh': (tanh, tanh_derivative),
    'logistic': (logistic, logistic_derivative),
    'sigmoid': (sigmoid, sigmoid_derivative),
    'softplus': (softplus, softplus_derivative),
}
