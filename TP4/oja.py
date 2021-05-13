import sys
from typing import Optional, Callable

import numpy as np
import pandas
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from TP3.config import Config
from TP3.config_to_network import _validate_base_network_params, _build_base_network_config
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


def biplot(score, coeff, pcax, pcay, labels=None):
    pca1 = pcax - 1
    pca2 = pcay - 1
    xs = score[:, pca1]
    ys = score[:, pca2]
    n = score.shape[1]
    scalex = 1.0 / (xs.max() - xs.min())
    scaley = 1.0 / (ys.max() - ys.min())
    plt.scatter(xs * scalex, ys * scaley)
    for i in range(n):
        plt.arrow(0, 0, coeff[i, pca1], coeff[i, pca2], color='r', alpha=0.5)
        if labels is None:
            plt.text(coeff[i, pca1] * 1.15, coeff[i, pca2] * 1.15, "Var" + str(i + 1), color='g', ha='center', va='center')
        else:
            plt.text(coeff[i, pca1] * 1.15, coeff[i, pca2] * 1.15, labels[i], color='g', ha='center', va='center')
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.xlabel("PC{}".format(pcax))
    plt.ylabel("PC{}".format(pcay))
    plt.grid()
    plt.show()


def oja(values: np.ndarray, eta: float, w: np.ndarray, epoch: int, tolerance: float) -> np.ndarray:
    iteration: int = 0
    for _ in range(epoch):
        for i in range(np.shape(values)[0]):
            iteration += 1
            sum: np.ndarray = np.dot(values[i], w)
            prev_w: np.ndarray = np.copy(w)
            w += eta * sum * (values[i] - sum * w)
            if np.linalg.norm(w - prev_w) < tolerance:
                print(iteration)
                return w / np.linalg.norm(w)
    return w / np.linalg.norm(w)


def exercise(config_file: str):
    europe = pandas.read_csv('../TP4/europe.csv')
    config: Config = Config(config_file)
    values = StandardScaler().fit_transform(europe.values[:, 1:])

    # TODO: No depender del config de TP3. Ver que parametros realmente varian y armar uno nuevo, mas simple
    network_config: NeuralNetworkBaseConfiguration = _build_base_network_config(
        _validate_base_network_params(config.network),
        len(values[0]) - 1
    )

    pca = PCA()
    pca.fit(values)

    neural_network: OjaNeuralNetwork = OjaNeuralNetwork(network_config)
    neural_network.train(values, np.zeros(1), None)
    print(neural_network.get_normalized_weights())
    print(neural_network.l_rate, neural_network.training_iteration)
    print(oja(values, 0.001, neural_network._perceptron.training_w, 150000, 1e-8))
    print(pca.components_[0])


if __name__ == "__main__":
    argv = sys.argv

    config_file: str = 'config.yaml'
    if len(argv) > 1:
        config_file = argv[1]

    exercise(config_file)


