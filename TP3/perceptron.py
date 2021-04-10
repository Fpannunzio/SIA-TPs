from collections import Callable

import numpy as np

Function = Callable[[float], float]


class Perceptron:
    def __init__(self, l_rate: float, activation_func: Function) -> None:
        super().__init__()
        self.l_rate = l_rate
        self.activation_func = activation_func

    def generate_hyperplane_coefficients(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return []

