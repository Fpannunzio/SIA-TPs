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
        previous_energy: float = self.calculate_energy(analyzed_value)
        next_sign: np.ndarray = np.sign(previous_sign @ self.weights)
        next_energy: float = self.calculate_energy(previous_sign)
        iterations: int = 0


        while not (np.array_equal(previous_sign, next_sign) or iterations > DEFAULT_MAX_ITERATIONS or previous_energy == next_energy):
            previous_sign = next_sign
            next_sign = np.sign(previous_sign @ self.weights)
            iterations += 1

        return next_sign

    def calculate_energy(self, analyzed_value: np.ndarray) -> float:
        energy: float = 0
        sign: np.ndarray = np.sign(analyzed_value @ self.weights)
        for i in range(np.size(self.weights, 1)):
            for j in range(np.size(self.weights, 1)):
                energy += self.weights[i][j] * sign[i] * sign[j]

        return energy / -2