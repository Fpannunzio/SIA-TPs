from typing import Dict, Callable, Tuple

import numpy as np
from schema import And, Schema, Or, Optional

from config import Param, Config
from perceptron import MultilayeredNeuralNetwork, StepNeuralNetwork, LinearNeuralNetwork, \
    NonLinearSinglePerceptronNeuralNetwork, ActivationFunction, NeuralNetwork

NeuralNetworkFactory = Callable[[float, int, Param], NeuralNetwork]
SigmoidFunction = Callable[[float, float], float]
SigmoidDerivativeFunction = SigmoidFunction


def _validate_perceptron_params(perceptron_params: Param) -> Param:
    return Config.validate_param(perceptron_params, Schema({
        'type': And(str, Or(*tuple(_perceptron_factory_map.keys()))),
        'learning_rate': And(Or(float, int), lambda lr: lr > 0),
        'momentum_factor': And(Or(float, int), lambda lr: lr > 0),
        'variable_learning_rate_factor': And(Or(float, int), lambda lr: lr > 0),
        Optional('params', default=dict): dict,
    }, ignore_extra_keys=True))


def get_perceptron(perceptron_params: Param, input_count: int) -> NeuralNetwork:
    _validate_perceptron_params(perceptron_params)

    factory: NeuralNetworkFactory = _perceptron_factory_map[perceptron_params['type']]

    return factory(perceptron_params['learning_rate'], input_count, perceptron_params['params'])


def _get_simple_perceptron(l_rate: float, input_count: int, params: Param) -> NeuralNetwork:
    return StepNeuralNetwork(l_rate, input_count)


def _get_linear_perceptron(l_rate: float, input_count: int, params: Param) -> NeuralNetwork:
    return LinearNeuralNetwork(l_rate, input_count)


def _validate_non_linear_perceptron_params(params: Param) -> Param:
    return Config.validate_param(params, Schema({
        'activation_function': And(str, Or(*tuple(_sigmoid_activation_function_map.keys()))),
        'slope_factor': And(Or(float, int), lambda b: b > 0)
    }, ignore_extra_keys=True))


def _get_non_linear_perceptron(l_rate: float, input_count: int, params: Param) -> NeuralNetwork:
    params = _validate_non_linear_perceptron_params(params)
    activation_function, activation_derivative = _sigmoid_activation_function_map[params['activation_function']]

    real_activation_function: ActivationFunction = lambda x: activation_function(x, params['slope_factor'])
    real_activation_derivative: ActivationFunction = lambda x: activation_derivative(x, params['slope_factor'])

    return NonLinearSinglePerceptronNeuralNetwork(l_rate, input_count, real_activation_function, real_activation_derivative)


def _validate_multi_layered_perceptron_params(params: Param) -> Param:
    return Config.validate_param(params, Schema({
        'activation_function': And(str, Or(*tuple(_sigmoid_activation_function_map.keys()))),
        'slope_factor': And(Or(float, int), lambda b: b > 0),
        'layer_sizes': And(list[int])  # TODO como verificar la lista que sean todos enteros positivos
    }, ignore_extra_keys=True))


def _get_multi_layered_perceptron(l_rate: float, input_count: int, params: Param) -> NeuralNetwork:
    params = _validate_multi_layered_perceptron_params(params)
    activation_function, activation_derivative = _sigmoid_activation_function_map[params['activation_function']]

    real_activation_function: ActivationFunction = lambda x: activation_function(x, params['slope_factor'])
    real_activation_derivative: ActivationFunction = lambda x: activation_derivative(x, params['slope_factor'])

    return MultilayeredNeuralNetwork(l_rate, input_count, real_activation_function, real_activation_derivative,
                                  params['layer_sizes'])


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


_perceptron_factory_map: Dict[str, NeuralNetworkFactory] = {
    'simple': _get_simple_perceptron,
    'linear': _get_linear_perceptron,
    'non_linear': _get_non_linear_perceptron,
    'multi_layered': _get_multi_layered_perceptron,
}

_sigmoid_activation_function_map: Dict[str, Tuple[SigmoidFunction, SigmoidDerivativeFunction]] = {
    'tanh': (tanh, tanh_derivative),
    'logistic': (logistic, logistic_derivative),
}
