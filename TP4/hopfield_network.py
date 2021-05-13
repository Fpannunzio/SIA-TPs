import numpy as np

DEFAULT_MAX_ITERATIONS: int = 10000


class HopfieldNetwork:

    def __init__(self, patterns: np.ndarray):
        self.weights: np.ndarray = self.initialize_hopfield_weights(patterns)

    @staticmethod
    def initialize_hopfield_weights(patterns: np.ndarray) -> np.ndarray:
        pattern_size: float = np.size(patterns, 1)
        # TODO(tobi): Por que se inicializa por 0?
        initial_weights: np.ndarray = np.zeros((pattern_size, pattern_size))

        # patterns = K.T => K @ K.T = patterns.T @ patterns
        initial_weights += (1 / pattern_size) * patterns.T @ patterns
        for i in range(int(pattern_size)):
            initial_weights[i][i] = 0

        return initial_weights

    def evaluate(self, analyzed_value: np.ndarray) -> np.ndarray:
        # TODO(tobi): Q - como viene analyzed value? Se pueden multiplicar?
        if np.size(analyzed_value) != np.size(self.weights, 0):
            raise ValueError("The dimension of the analyzed value is incorrect")

        previous_sign: np.ndarray = np.sign(analyzed_value @ self.weights)
        next_sign: np.ndarray = np.sign(previous_sign @ self.weights)
        iterations: int = 0

        # TODO(tobi): Cortar tambien si el nivel de energia no baja por n iteraciones
        while not (np.array_equal(previous_sign, next_sign) or iterations > DEFAULT_MAX_ITERATIONS):
            previous_sign = next_sign
            next_sign = np.sign(previous_sign @ self.weights)
            iterations += 1

        return next_sign

    # TODO(tobi): Graficar energia
