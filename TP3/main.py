import sys
from typing import List

import numpy as np
import pandas as pd

from neural_network_utils import get_neural_network
from plot import plot_error
from config import Config, Param
from neural_network import NeuralNetwork


def get_training_set(file_name: str, line_count: int, normalize: bool) -> np.ndarray:
    training_set: np.ndarray = pd.read_csv(file_name, delim_whitespace=True, header=None).values
    if normalize:
        training_set = training_set / 100

    if line_count > 1:
        elem_size: int = len(training_set[0]) * line_count
        training_set = np.reshape(training_set, (np.size(training_set) // elem_size, elem_size))

    return training_set


def main(config_file: str):
    print(f'Loading config file {config_file}...')
    config: Config = Config(config_file)

    training_set: Param = config.training_set

    training_points: np.ndarray = get_training_set(training_set['inputs'], training_set['input_line_count'], training_set['normalize_values'])

    training_values: np.ndarray = get_training_set(training_set['outputs'], training_set['output_line_count'], training_set['normalize_values'])

    neural_network: NeuralNetwork = get_neural_network(config.network, len(training_points[0]))

    network_error_by_iteration: List[float] = []

    def get_network_error(network: NeuralNetwork) -> None:
        network_error_by_iteration.append(network.error)
        print(network.error)
        print(network.training_iteration)

    neural_network.train(training_points, training_values, get_network_error)

    plot_error(network_error_by_iteration)

    print(neural_network.l_rate, neural_network.error, neural_network.training_iteration)
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

    try:
        main(config_file)

    except KeyboardInterrupt:
        sys.exit(0)

    except (ValueError, FileNotFoundError) as ex:
        print('\nAn Error Was Found!!')
        print(ex)
        sys.exit(1)

    except Exception as ex:
        print('An unexpected error occurred')
        raise ex
