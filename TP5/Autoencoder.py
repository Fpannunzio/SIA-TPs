from typing import List, Optional, Callable

import numpy as np
from attr import attr

from TP3.neural_network_lib.neural_network import MultilayeredNeuralNetwork, ActivationFunction, \
    NeuralNetworkErrorFunction, DEFAULT_MAX_ITERATIONS, DEFAULT_ERROR_TOLERANCE, DEFAULT_LINEAR_SEARCH_MAX_VALUE, \
    DEFAULT_L_RATE_LINEAR_SEARCH_MAX_ITERATIONS, DEFAULT_L_RATE_LINEAR_SEARCH_ERROR_TOLERANCE, \
    NeuralNetworkBaseConfiguration, NeuralNetwork


class Autoencoder:

    def __init__(self, network: MultilayeredNeuralNetwork) -> None:

        self.autoencoder: MultilayeredNeuralNetwork = network
        self.latent_layer = None

    def train(self, training_points: np.ndarray) -> None:
        self.autoencoder.train(training_points, training_points)


class VAE:
    def _validate_config(self, base_config: NeuralNetworkBaseConfiguration) -> bool:
        return (
            base_config.output_count is None or base_config.input_count == base_config.output_count
        )

    def _validate_training_data(self, training_points: np.ndarray) -> None:
        valid: bool = (
                len(training_points) > 0 and
                len(training_points[0]) == self.encoder.input_count
        )
        if not valid:
            raise ValueError(f'Invalid training data, doesnt match config.\n'
                             f'Network Input/Output: {repr(self.encoder.input_count)}\n'
                             f'Points: {repr(training_points)}\n')

    def __init__(self, base_config: NeuralNetworkBaseConfiguration,
                 activation_derivative: ActivationFunction,
                 error_function: NeuralNetworkErrorFunction,
                 layer_sizes: List[int]):
        if not self._validate_config(base_config):
            raise ValueError('VAE: Invalid configuration')

        self.error = np.inf

        self.latent_dim: int = layer_sizes[-1]

        # Build real layers
        encoder_layers: List[int] = layer_sizes.copy()
        encoder_layers[-1] = 2 * self.latent_dim

        decoder_layers: List[int] = list(reversed(layer_sizes))

        self.mean: np.ndarray
        self.sigma: np.ndarray
        self.epsilon: np.ndarray

        self.encoder: MultilayeredNeuralNetwork = MultilayeredNeuralNetwork(base_config, activation_derivative, NeuralNetworkErrorFunction.IDENTITY, encoder_layers)

        base_config.input_count = self.latent_dim
        self.decoder: MultilayeredNeuralNetwork = MultilayeredNeuralNetwork(base_config, activation_derivative, error_function, decoder_layers)

    def encode(self, point: np.ndarray, training: bool = False, insert_identity_column: bool = False):
        mean_epsilon: np.ndarray = self.encoder.predict(point, training, insert_identity_column)
        return np.random.normal(0, 1, size=self.latent_dim) * mean_epsilon[:self.latent_dim] + mean_epsilon[self.latent_dim:]

    def decode(self, point: np.ndarray, training: bool = False, insert_identity_column: bool = False) -> np.ndarray:
        return self.decoder.predict(point, training, insert_identity_column)

    def encode_points(self, points: np.ndarray, training: bool = False, insert_identity_column: bool = False) -> np.ndarray:
        if insert_identity_column:
            points = NeuralNetwork.with_identity_dimension(points)
        return np.apply_along_axis(self.encode, 1, points, training)

    def generate(self) -> np.ndarray:
        return self.decode(np.random.normal(self.mean, self.sigma, size=self.latent_dim), training=False, insert_identity_column=True)

    # Deben venir SIN la identity column
    def calculate_error(self, points: np.ndarray, training: bool = False) -> float:
        return (
            self.decoder.calculate_error(self.encode_points(points, training, insert_identity_column=True), points, training, insert_identity_column=True)
            - 0.5 * (self.latent_dim + np.sum(self.sigma - np.square(self.mean) - np.exp(self.sigma)))
        )

    # Main training function
    def train(self, training_points: np.ndarray, status_callback: Optional[Callable[['VAE', int], None]] = None,
              insert_identity_column: bool = True) -> None:

        training_values: np.ndarray = training_points

        self._validate_training_data(training_points)

        if insert_identity_column:
            training_points = NeuralNetwork.with_identity_dimension(training_points)

        while not (self.encoder.has_training_ended() or self.decoder.has_training_ended()):

            self.encoder.check_soft_reset(training_points)
            self.decoder.check_soft_reset(training_points)

            # Select point to train
            selected_point: int = self.encoder.choose_point(training_points)

            # Calculo parametros de la normal
            merged_mu_sigma: np.ndarray = self.encoder.predict(training_points[selected_point], training=True)

            # Decido arbitrariamente que la primera mitad es mu y la segunda sigma
            self.mean = merged_mu_sigma[:self.latent_dim]
            self.sigma = merged_mu_sigma[self.latent_dim:]

            # Calculo un epsilon arbirario con distribucion gaussiana
            self.epsilon = np.random.normal(0, 1, size=self.latent_dim)

            # Sampleo z con los parametros obtenidos
            z: np.ndarray = self.mean + self.sigma * self.epsilon
            z = NeuralNetwork.with_identity_dimension(z, axis=0)

            # Update direction to use on weight update
            x_hat: np.ndarray = self.decoder.predict(z, training=True)

            # Update direction to use on weight update
            last_decoder_layer_delta: np.ndarray = self.decoder.update_deltas(self.decoder.calculate_first_delta(training_values[selected_point], x_hat))

            # LINEAR SEARCH NOT SUPPORTED
            # if self.decoder.linear_search_l_rate:
            #    self.decoder.recalculate_l_rate_with_linear_search(training_points, training_values)

            # Update weight using current l_rate and delta direction
            self.decoder.update_training_weight(z)

            # Calculate error
            current_error: float = self.calculate_error(training_values, training=True)

            self.decoder.update_state(current_error)

            # Calculo los deltas de la ultima capa del encoder
            mean_delta: np.ndarray = last_decoder_layer_delta + self.mean
            sigma_delta: np.ndarray = last_decoder_layer_delta * self.epsilon - 0.5 * (1 - np.exp(self.sigma))

            # Calculo los deltas del encoder
            self.encoder.update_deltas(np.concatenate((mean_delta, sigma_delta)))

            # LINEAR SEARCH NOT SUPPORTED
            # if self.encoder.linear_search_l_rate:
            #    self.encoder.recalculate_l_rate_with_linear_search(training_points, training_values)

            # Update weight using current l_rate and delta direction
            self.encoder.update_training_weight(training_points[selected_point])

            # TODO(tobi): Ver que onda este error. En principio no afecta mucho, pero se puede calcular bien
            self.encoder.update_state(current_error)

            has_error_improved: bool = current_error < self.error
            if has_error_improved:
                self.error = current_error

            if status_callback:
                status_callback(self, selected_point)
