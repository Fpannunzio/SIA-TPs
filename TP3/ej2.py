import sys
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

from plot import plot_error
from neural_network_utils import NeuralNetworkFactory, cross_validation, CrossValidationResult
from config import Param, Config
from config_to_network import get_neural_network, get_neural_network_factory, get_training_set
from neural_network import NeuralNetwork


def generate_config() -> Param:
    network_params: Param = {}

    network_params['max_training_iterations'] = 10000
    network_params['weight_reset_threshold'] = network_params['max_training_iterations']
    network_params['max_stale_error_iterations'] = network_params['max_training_iterations']
    network_params['error_goal'] = 0.01
    network_params['error_tolerance'] = 0.01
    network_params['momentum_factor'] = 0.5
    network_params['base_learning_rate'] = None
    network_params['learning_rate_strategy'] = None

    # Learning Rate Linear Search Params
    network_params['learning_rate_linear_search_params'] = {}
    l_rate_linear_search_params: Param = network_params['learning_rate_linear_search_params']

    l_rate_linear_search_params['max_iterations'] = 1000
    l_rate_linear_search_params['max_value'] = 1
    l_rate_linear_search_params['error_tolerance'] = network_params['error_tolerance']

    # Variable Learning Rate Params
    network_params['variable_learning_rate_params'] = {}
    variable_l_rate_params: Param = network_params['variable_learning_rate_params']

    variable_l_rate_params['down_scaling_factor'] = 0.1
    variable_l_rate_params['up_scaling_factor'] = 0.1 # Cuando se use lo setea cada uno
    variable_l_rate_params['positive_trend_threshold'] = 10
    variable_l_rate_params['negative_trend_threshold'] = variable_l_rate_params['positive_trend_threshold'] * 50

    # Network params params
    network_params['network_params'] = {}
    network_params_params: Param = network_params['network_params']
    network_params_params['activation_function'] = 'tanh'
    network_params_params['activation_slope_factor'] = 0.6
    network_params_params['error_function'] = 'quadratic'

    return network_params

# https://stackoverflow.com/a/49601444/12270520
def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])

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