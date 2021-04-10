import random
from collections import Callable
from typing import Type, Dict

import numpy as np
from schema import Schema, And, Or

from TP3.config import Param, Config

Function = Callable[[float], float]


class Perceptron:
    def __init__(self, l_rate: float) -> None:
        super().__init__()
        self.l_rate = l_rate
        self.activation_func = np.sign

    def heavy_sum(self, x: np.ndarray, w: np.array) -> float:
        return np.sum(x * w)

    def calculate_delta(self, x: np.ndarray, y: float, ac: float) -> np.ndarray:
        return x * self.l_rate * (y - ac)

    def generate_hyperplane_coefficients(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        x = np.insert(x, 0, 1, axis=1)
        w: np.array = np.random.rand(len(x[0])) * 2 - 1
        w_min: np.ndarray = np.copy(w)
        error: float = 1.0
        error_min: float = len(x) * 2
        i: int = 0
        n: int = 0

        while error > 0 and i < 10000:
            if n > 1000 * len(x):
                w = np.random.rand(len(x[0])) * 2 - 1
                n = 0
            i_x: int = random.randint(0, len(x) - 1)
            ac: float = self.activation_func(self.heavy_sum(x[i_x], w))
            w += self.calculate_delta(x[i_x], y[i_x], ac)
            error = self.calculate_error(x, y, w)

            if error < error_min:
                error_min = error
                w_min = np.copy(w)
            i += 1

        for j in range(len(x)):
            print(f'Expected value {y[j]} got: {self.activation_func(self.heavy_sum(x[j], w))}')
        return w

    def calculate_error(self, x: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
        return sum([0.5 * (abs(y[j] - self.activation_func(self.heavy_sum(x[j], w)))) ** 2 for j in range(len(x))])


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
    learning_rate = perceptron_params['learning_rate']

    return p_type(learning_rate)
