import sys
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

from TP3.config import Param
from TP3.config_to_network import get_neural_network
from TP3.exercises_utils import get_training_set, generate_config, lighten_color
from TP3.neural_network import NeuralNetwork



class EJ2:

    def __init__(self) -> None:
        self.training_points: np.ndarray = get_training_set('trainingset/inputs/Ej2.tsv', 1, True)
        self.training_values: np.ndarray = get_training_set('trainingset/outputs/Ej2.tsv', 1, True)

        self.predictions: Dict[str, Dict[str, List[float]]] = {}
        self.min_error: List[float] = []
        self.last_error: List[float] = []

        self.network_params: Param = generate_config()

    def get_network_error(self, network: NeuralNetwork, selected_training_point: int) -> None:
        self.min_error.append(network.error)
        self.last_error.append(network.last_training_error)

    def linear(self):
        self.network_params['type'] = 'linear'
        neural_network: NeuralNetwork

        self.predictions['linear'] = {}

        # Small
        self.network_params['learning_rate_strategy'] = 'fixed'
        self.network_params['base_learning_rate'] = 0.00001
        # self.network_params['variable_learning_rate_params']['up_scaling_factor'] = \
        #     self.network_params['variable_learning_rate_params']['down_scaling_factor'] \
        #     * self.network_params['base_learning_rate']
        neural_network = get_neural_network(self.network_params, len(self.training_points[0]))
        self.min_error = []
        self.last_error = []

        neural_network.train(self.training_points, self.training_values, self.get_network_error)

        self.predictions['linear']['small_min'] = self.min_error
        self.predictions['linear']['small_last'] = self.last_error

        # Medium
        self.network_params['learning_rate_strategy'] = 'fixed'
        self.network_params['base_learning_rate'] = 0.05
        neural_network = get_neural_network(self.network_params, len(self.training_points[0]))
        self.min_error = []
        self.last_error = []

        neural_network.train(self.training_points, self.training_values, self.get_network_error)

        self.predictions['linear']['medium_min'] = self.min_error
        self.predictions['linear']['medium_last'] = self.last_error

        # Big
        self.network_params['learning_rate_strategy'] = 'fixed'
        self.network_params['base_learning_rate'] = 0.8
        neural_network = get_neural_network(self.network_params, len(self.training_points[0]))
        self.min_error = []
        self.last_error = []

        neural_network.train(self.training_points, self.training_values, self.get_network_error)

        self.predictions['linear']['big_min'] = self.min_error
        self.predictions['linear']['big_last'] = self.last_error

        plt.plot(self.predictions['linear']['big_last'], color=lighten_color('g', 0.3),  label='big_last')
        plt.plot(self.predictions['linear']['medium_last'], color=lighten_color('m', 0.3), label='medium_last')
        plt.plot(self.predictions['linear']['small_last'], color=lighten_color('k', 0.3), label='small_last')
        plt.plot(self.predictions['linear']['big_min'], 'g-', label='big_min', lw=2)
        plt.plot(self.predictions['linear']['medium_min'], 'm-', label='medium_min', lw=2)
        plt.plot(self.predictions['linear']['small_min'], 'k-', label='small_min', lw=2)
        plt.legend()

        plt.semilogy()
        plt.title('loss per model - momentum: 0.5')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.show()


def ej2(config_file: str):
    print(f'Loading config file {config_file}...')
    config: Config = Config(config_file)

    training_set: Param = config.training_set

    training_points: np.ndarray = get_training_set(training_set['inputs'], training_set['input_line_count'], training_set['normalize_values'])

    training_values: np.ndarray = get_training_set(training_set['outputs'], training_set['output_line_count'], training_set['normalize_values'])

    neural_network_factory: NeuralNetworkFactory = get_neural_network_factory(config.network, len(training_points[0]))

    def error_metric(nn: NeuralNetwork, points: np.ndarray, values: np.ndarray) -> float:
        return nn.calculate_error(points, values, training=False, insert_identity_column=True)

    validation_result: CrossValidationResult = cross_validation(neural_network_factory, training_points, training_values,
                                                                error_metric, len(training_points)//10, 1)

    print(validation_result)

    best_neural_network: NeuralNetwork = validation_result.best_neural_network

    best_points: np.ndarray = validation_result.best_test_points
    best_values: np.ndarray = validation_result.best_test_values

    error_count: int = len(best_neural_network.validate_points(best_points, best_values, error_tolerance=0.001))
    close_count: int = len(best_values) - error_count

    print(f'Correct Values: {close_count}\nError Values:{error_count}')

    neural_network2: NeuralNetwork = neural_network_factory()

    errors_over_iter: List[float] = []

    def error_over_iterations(neural_network: NeuralNetwork, point: int):
        errors_over_iter.append(neural_network.error)

    neural_network2.train(best_points, best_values, error_over_iterations)

    plot_error(errors_over_iter)


if __name__ == '__main__':
    argv = sys.argv

    config_file: str = 'config.yaml'
    if len(argv) > 1:
        config_file = argv[1]

    try:
        ej2(config_file)

    except KeyboardInterrupt:
        sys.exit(0)

    except (ValueError, FileNotFoundError) as ex:
        print('\nAn Error Was Found!!')
        print(ex)
        sys.exit(1)

    except Exception as ex:
        print('An unexpected error occurred')
        raise ex