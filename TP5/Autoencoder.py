from typing import List

import numpy as np

from TP3.neural_network_lib.neural_network import MultilayeredNeuralNetwork


class Autoencoder:

    def __init__(self, network: MultilayeredNeuralNetwork) -> None:

        self.autoencoder: MultilayeredNeuralNetwork = network
        self.latent_layer = None

    def train(self, training_points: np.ndarray) -> None:
        self.autoencoder.train(training_points, training_points)

