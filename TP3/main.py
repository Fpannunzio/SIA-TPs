import sys
from typing import Dict

import numpy as np
import pandas as pd

# from perceptron_utils import get_perceptron
from config import Config
from perceptron import NeuralNetwork, MultilayeredNeuralNetwork


def main(config_file: str):
    print(f'Loading config file {config_file}...')
    config: Config = Config(config_file)

    training_set: Dict[str, str] = config.training_set

    training_points: np.ndarray = pd.read_csv(training_set['inputs'], delim_whitespace=True, header=None).values
    training_values: np.ndarray = pd.read_csv(training_set['outputs'], delim_whitespace=True, header=None).values
      # Turn n x 1 matrix into array with length n

    nn: NeuralNetwork = MultilayeredNeuralNetwork(0.1, 2, lambda x: np.tanh(0.6 * x), lambda x: 0.6 * (1 - np.tanh(x * 0.6) ** 2), [4, 1])

    nn.train(training_points, training_values)
    # Get Perceptron according to config, and as many inputs as training points dimension
    # perceptron: Perceptron = get_perceptron(config.perceptron, len(training_points[0]))
    #
    # # Train Perceptron with training data!
    # iteration_count, last_w = perceptron.train(training_points, training_values)
    #
    # print(iteration_count)
    # print(last_w)
    # print(perceptron.w)
    # print(perceptron.error)
    #
    # if len(training_points[0]) == 2:
    #     plot_2d_hyperplane(training_points, training_values, perceptron.w)
    #
    # validation_set: Dict[str, str] = config.validation_set
    #
    # validation_points: np.ndarray = pd.read_csv(validation_set['inputs'], delim_whitespace=True, header=None).values
    # validation_values: np.ndarray = pd.read_csv(validation_set['outputs'], delim_whitespace=True, header=None).values
    # validation_values = np.squeeze(validation_values)  # Turn n x 1 matrix into array with length n
    #
    # # Validate Perceptron with Validation Points
    # failed_points: np.ndarray = perceptron.validate_points(validation_points, validation_values)
    #
    # if len(failed_points) > 0:
    #     print('La solucion encontrada no paso la prueba de validacion. No pude precedir correctamente el valor de los siguientes puntos:')
    #     print(failed_points)
    # else:
    #     print('La solucion encontrada paso la prueba de validacion')


if __name__ == "__main__":
    argv = sys.argv

    config_file: str = 'config.yaml'
    if len(argv) > 1:
        config_file = argv[1]

    # try:
    main(config_file)

    # except KeyboardInterrupt:
    #     sys.exit(0)
    #
    # except (ValueError, FileNotFoundError) as ex:
    #     print('\nAn Error Was Found!!')
    #     print(ex)
    #     sys.exit(1)
    #
    # except Exception as ex:
    #     print('An unexpected error occurred')
    #     raise ex
