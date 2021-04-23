import sys
from typing import List

import numpy as np
import pandas as pd

from neural_network_utils import get_neural_network
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

    # cross_validation(config.network, training_points, training_values, _neural_network_metrics['error'], 10, 10)

    network_error_by_iteration: List[float] = []

    def get_network_error(network: NeuralNetwork, selected_training_point: int) -> None:
        network_error_by_iteration.append(network.error)
        print(network.l_rate, network.error, network.last_training_error, network.training_iteration)

    neural_network.train(training_points, training_values, get_network_error)

    # plot_error(network_error_by_iteration)


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
