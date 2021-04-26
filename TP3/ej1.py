import sys
from typing import List

import numpy as np

from config import Config, Param
from config_to_network import get_training_set, get_neural_network
from neural_network import NeuralNetwork, SinglePerceptronNeuralNetwork
from plot import plot_error, plot_2d_hyperplane


def ej1(config_file: str):

    print(f'Loading config file {config_file}...')
    config: Config = Config(config_file)

    training_set: Param = config.training_set

    # Defaults
    if not training_set or training_set['inputs'] is None:
        training_set['inputs'] = 'trainingset/inputs/Ej1-AND.tsv'
    if not training_set or training_set['outputs'] is None:
        training_set['outputs'] = 'trainingset/outputs/Ej1-AND.tsv'
    if not training_set or training_set['normalize_values'] is None:
        training_set['normalize_values'] = False

    training_points: np.ndarray = get_training_set(training_set['inputs'], training_set['input_line_count'], training_set['normalize_values'])

    training_values: np.ndarray = get_training_set(training_set['outputs'], training_set['output_line_count'], training_set['normalize_values'])

    neural_network: NeuralNetwork = get_neural_network(config.network, len(training_points[0]))

    network_error_by_iteration: List[float] = []

    def get_network_error(network: NeuralNetwork, selected_training_point: int) -> None:
        network_error_by_iteration.append(network.error)

    neural_network.train(training_points, training_values, get_network_error)

    if config.plot:
        plot_error(network_error_by_iteration)

    if isinstance(neural_network, SinglePerceptronNeuralNetwork):
        print(f'Perceptron Weights: {neural_network._perceptron.w}')
        if len(training_points[0]) == 2 and len(training_values[0]) == 1 and config.plot:
            plot_2d_hyperplane(training_points, training_values, neural_network._perceptron.w)
    else:
        print('ej1.py is better suited for a single perceptron neural networks. Consider using it next time!')


if __name__ == "__main__":
    argv = sys.argv

    config_file: str = 'config.yaml'
    if len(argv) > 1:
        config_file = argv[1]

    try:
        ej1(config_file)

    except KeyboardInterrupt:
        sys.exit(0)

    except (ValueError, FileNotFoundError) as ex:
        print('\nAn Error Was Found!!')
        print(ex)
        sys.exit(1)

    except Exception as ex:
        print('An unexpected error occurred')
        raise ex
