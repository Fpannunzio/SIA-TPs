from typing import Dict, Callable, Tuple

import numpy as np
from schema import And, Schema, Or, Optional

from perceptron import Perceptron, SimplePerceptron, LinearPerceptron, NonLinearPerceptron, ActivationFunction
from config import Param, Config

PerceptronFactory = Callable[[float, np.ndarray, np.ndarray, Param], Perceptron]
SigmoidFunction = Callable[[float, float], float]
SigmoidDerivativeFunction = SigmoidFunction


def _validate_perceptron_params(perceptron_params: Param) -> Param:
    return Config.validate_param(perceptron_params, Schema({
        'type': And(str, Or(*tuple(_perceptron_factory_map.keys()))),
        'learning_rate': And(Or(float, int), lambda lr: lr > 0),
        Optional('params', default=dict): dict,
    }, ignore_extra_keys=True))


def get_perceptron(perceptron_params: Param, training_points: np.ndarray, training_values: np.ndarray) -> Perceptron:
    _validate_perceptron_params(perceptron_params)

    factory: PerceptronFactory = _perceptron_factory_map[perceptron_params['type']]

    return factory(perceptron_params['learning_rate'], training_points, training_values, perceptron_params['params'])


def _get_simple_perceptron(l_rate: float, training_points: np.ndarray, training_values: np.ndarray, params: Param) -> Perceptron:
    return SimplePerceptron(l_rate, training_points, training_values)


def _get_linear_perceptron(l_rate: float, training_points: np.ndarray, training_values: np.ndarray, params: Param) -> Perceptron:
    return LinearPerceptron(l_rate, training_points, training_values)


def _validate_non_linear_perceptron_params(params: Param) -> Param:
    return Config.validate_param(params, Schema({
        'activation_function': And(str, Or(*tuple(_sigmoid_activation_function_map.keys()))),
        'slope_factor':        And(Or(float, int), lambda b: b > 0)
    }, ignore_extra_keys=True))


def _get_non_linear_perceptron(l_rate: float, training_points: np.ndarray, training_values: np.ndarray, params: Param) -> Perceptron:
    params = _validate_non_linear_perceptron_params(params)
    activation_function, activation_derivative = _sigmoid_activation_function_map[params['activation_function']]

    real_activation_function:   ActivationFunction = lambda x: activation_function(x, params['slope_factor'])
    real_activation_derivative: ActivationFunction = lambda x: activation_derivative(x, params['slope_factor'])

    return NonLinearPerceptron(l_rate, real_activation_function, real_activation_derivative, training_points, training_values)


# Sigmoid Activation Functions And Derivatives

# Hyperbolic Tangent
def tanh(x: float, b: float) -> float:
    return np.tanh(b*x)


def tanh_derivative(x: float, b: float) -> float:
    return b*(1 - tanh(x, b) ** 2)


# Logistic Function
def logistic(x: float, b: float) -> float:
    return 1 / (1 + np.exp(-2*b*x))


def logistic_derivative(x: float, b: float) -> float:
    return 2*b*logistic(x, b)*(1 - logistic(x, b))


_perceptron_factory_map: Dict[str, PerceptronFactory] = {
    'simple':       _get_simple_perceptron,
    'linear':       _get_linear_perceptron,
    'non_linear':   _get_non_linear_perceptron,
}

_sigmoid_activation_function_map: Dict[str, Tuple[SigmoidFunction, SigmoidDerivativeFunction]] = {
    'tanh': (tanh, tanh_derivative),
    'logistic': (logistic, logistic_derivative),
}
