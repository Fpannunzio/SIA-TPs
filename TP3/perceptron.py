import math
import random
from collections import Callable
from typing import Type, Dict

import numpy as np
from schema import Schema, And, Or

from config import Param, Config
from plot import AsyncPlotter

Function = Callable[[float], float]


def calculate_heavy_sum(x: np.ndarray, w: np.array) -> float:
    return np.sum(x * w)


class Perceptron:
    function_map: Dict[str, Function] = {
        'sign': np.sign
    }

    def __init__(self, params: Param) -> None:
        super().__init__()
        self._validate_params(params)
        self.l_rate = params['learning_rate']
        self.activation_func = self.function_map[params['function']]
        self.w_min: np.array = None

    def calculate_delta(self, x: np.ndarray, y: float, heavy_sum: float) -> np.ndarray:
        return x * self.l_rate * (y - self.activation_func(heavy_sum))

    def generate_hyperplane_coefficients(self, x: np.ndarray, y: np.ndarray, plotter: AsyncPlotter) -> np.ndarray:
        x = np.insert(x, 0, 1, axis=1)
        w: np.array = np.random.rand(len(x[0])) * 2 - 1
        self.w_min: np.ndarray = np.copy(w)
        error: float = 1.0
        error_min: float = len(x) * 2
        i: int = 0
        n: int = 0

        while error > 0 and i < 10000:
            if n > 1000 * len(x):
                w = np.random.rand(len(x[0])) * 2 - 1
                n = 0
            i_x: int = random.randint(0, len(x) - 1)
            w += self.calculate_delta(x[i_x], y[i_x], calculate_heavy_sum(x[i_x], w))
            error = self.calculate_error(x, y, w)

            if error < error_min:
                error_min = error
                self.w_min = np.copy(w)
                plotter.publish(self.w_min)
            i += 1

        return w

    def are_validate_coefficients_valid(self, x: np.ndarray, y: np.ndarray) -> bool:
        x = np.insert(x, 0, 1, axis=1)
        for i in range(len(x)):
            if not math.isclose(self.activation_func(calculate_heavy_sum(x[i], self.w_min)), y[i]):
                return False
        return True

    def calculate_error(self, x: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
        return sum(abs(y[j] - self.activation_func(calculate_heavy_sum(x[j], w))) for j in range(len(x)))

    def _validate_params(self, params):
        pass


class LinearPerceptron(Perceptron):
    function_map: Dict[str, Function] = {
        'linear': lambda var: var,
    }

    def calculate_error(self, x: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
        return sum([0.5 * (abs(y[j] - self.activation_func(calculate_heavy_sum(x[j], w)))) ** 2 for j in range(len(x))])


class NonLinearPerceptron(LinearPerceptron):
    function_map: Dict[str, Function] = {
        'tanh': np.tanh,
    }

    derivative_map: Dict[str, Function] = {
        'tanh': lambda var: 1 - np.tanh(var) ** 2
    }

    def __init__(self, params: Param) -> None:
        super().__init__(params)
        self.derivative = self.derivative_map[params['function']]

    def calculate_delta(self, x: np.ndarray, y: float, heavy_sum: float) -> np.ndarray:
        return super().calculate_delta(x, y, heavy_sum) * self.derivative(heavy_sum)


def _validate_perceptron(perceptron_params: Param) -> Param:
    return Config.validate_param(perceptron_params, Schema({
        'type': And(str, Or(*tuple(_perceptron_map.keys()))),
        'learning_rate': And(Or(float, int), lambda lr: lr > 0)
    }, ignore_extra_keys=True))


_perceptron_map: Dict[str, Type[Perceptron]] = {
    'step': Perceptron
}


def get_perceptron(perceptron_params: Param) -> Perceptron:
    _validate_perceptron(perceptron_params)

    p_type: Type[Perceptron] = _perceptron_map[perceptron_params['type']]

    return p_type(perceptron_params)
