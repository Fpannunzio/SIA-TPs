import math
from typing import List, Tuple

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

    def evaluate(self, analyzed_value: np.ndarray) -> Tuple[np.ndarray, List[float]]:
        # TODO(tobi): Q - como viene analyzed value? Se pueden multiplicar?
        if np.size(analyzed_value) != np.size(self.weights, 0):
            raise ValueError("The dimension of the analyzed value is incorrect")

        previous_sign: np.ndarray = np.sign(analyzed_value @ self.weights)
        previous_energy: float = self.calculate_energy(analyzed_value)
        next_sign: np.ndarray = np.sign(previous_sign @ self.weights)
        next_energy: float = self.calculate_energy(previous_sign)
        iterations: int = 0
        energies: List[float] = []
        similar_energies: List[float] = []
        osilating_cache: List[float] = [previous_energy, next_energy]
        osilating: int = 0

        energies.append(previous_energy)
        energies.append(next_energy)

        if previous_energy == next_energy:
            similar_energies.append(previous_energy)
        similar_energies.append(next_energy)

        while not (np.array_equal(previous_sign, next_sign) or iterations > DEFAULT_MAX_ITERATIONS or len(similar_energies) >= 3) or osilating >= 3:
            previous_sign = next_sign
            next_sign = np.sign(previous_sign @ self.weights)
            iterations += 1
            previous_energy = next_energy
            next_energy = self.calculate_energy(previous_sign)
            if previous_energy != next_energy:
                similar_energies.clear()
            similar_energies.append(next_energy)
            energies.append(next_energy)

            if math.isclose(next_energy, osilating_cache[0]):
                osilating += 1
            else:
                osilating = 0

            osilating_cache[2] = next_energy
            osilating_cache.pop(0)

        return next_sign, energies

    def calculate_energy(self, s: np.ndarray) -> float:
        energy: float = 0
        for i in range(np.size(self.weights, 1)):
            for j in range(np.size(self.weights, 1)):
                energy += self.weights[i][j] * s[i] * s[j]

        return energy / -2