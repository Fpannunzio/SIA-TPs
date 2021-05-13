from typing import Optional, Callable

import numpy as np
from TP3.neural_network_lib.neural_network import NeuralNetwork, SinglePerceptronNeuralNetwork, Perceptron, \
    NeuralNetworkBaseConfiguration, NeuralNetworkErrorFunction


class OjaPerceptron(Perceptron):

    # delta_w = eta(Y * X_n - Y^2 * wj)
    def update_training_weights(self, l_rate: float, point: np.ndarray) -> None:
        new_delta_w: np.ndarray = l_rate * self.last_excitement * (point - self.last_excitement * self.training_w)
        self.training_w += new_delta_w
        self._last_delta_w = new_delta_w

    def normalized_weight(self) -> float:
        return self.training_w / np.linalg.norm(self.training_w)


class OjaNeuralNetwork(SinglePerceptronNeuralNetwork):

    def __init__(self, base_config: NeuralNetworkBaseConfiguration) -> None:
        base_config.activation_fn = lambda x: x
        base_config.error_function = NeuralNetworkErrorFunction.ABSOLUTE
        super().__init__(base_config)

    def _perceptron_instance(self):
        return OjaPerceptron(self.input_count, self.activation_fn, self.momentum_factor)

    def train(self, training_points: np.ndarray, training_values: np.ndarray, status_callback: Optional[Callable[[NeuralNetwork, int], None]] = None,
              insert_identity_column: bool = True) -> None:

        # falta el validate training data
        # TODO(tobi): No deberiamos calcular el error y actualizar eta???
        for _ in range(self.max_training_iterations):
            for i in range(np.shape(training_points)[0]):
                self.predict(training_points[i])
                self._update_training_weight(training_points[i])

            # Una iteracion es una epoca
            self.training_iteration += 1
            if status_callback:
                status_callback(self, 0)

    def _calculate_delta(self, training_value: float, activation: float) -> float:
        pass

    def get_normalized_weights(self) -> float:
        return self._perceptron.normalized_weight()