import numpy as np

DEFAULT_MAX_ITERATIONS: int = 10000

class HopfieldNetwork:

    def __init__(self, patterns: np.ndarray):
        self.weights: np.ndarray = self.initialize_hopfield_weights(patterns)

    @staticmethod
    def initialize_hopfield_weights(patterns: np.ndarray) -> np.ndarray:
        pattern_size: float = np.size(patterns, 1)
        initial_weights: np.ndarray = np.zeros((pattern_size, pattern_size))

        initial_weights += (1 / pattern_size) * np.dot(patterns.T, patterns)
        for i in range(int(pattern_size)):
            initial_weights[i][i] = 0

        return initial_weights

    def evaluate(self, analized_value: np.ndarray) -> np.ndarray:
        if np.size(analized_value) != np.size(self.weights, 0):
            raise ValueError("The dimension of the analized value is incorrect")

        previous_sign: np.ndarray = np.sign(np.dot(analized_value, self.weights))
        next_sign: np.ndarray = np.sign(np.dot(previous_sign, self.weights))
        iterations: int = 0

        while not (np.array_equal(previous_sign, next_sign) or iterations > DEFAULT_MAX_ITERATIONS):
            previous_sign = next_sign
            next_sign = np.sign(np.dot(previous_sign, self.weights))
            iterations += 1

        return next_sign
