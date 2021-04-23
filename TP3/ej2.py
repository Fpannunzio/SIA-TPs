from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from TP3.config import Param
from TP3.config_to_network import get_neural_network, get_neural_network_factory, _error
from TP3.neural_network import NeuralNetwork
from TP3.neural_network_utils import cross_validation


def generate_config() -> Param:
    network_params: Param = {}

    network_params['max_training_iterations'] = 1000
    network_params['weight_reset_threshold'] = network_params['max_training_iterations']
    network_params['max_stale_error_iterations'] = network_params['max_training_iterations']
    network_params['error_goal'] = 0.01
    network_params['error_tolerance'] = 0.01
    network_params['momentum_factor'] = 0.8
    network_params['base_learning_rate'] = 0.8
    network_params['learning_rate_strategy'] = 'fixed'
    network_params['type'] = 'linear'

    # Learning Rate Linear Search Params
    network_params['learning_rate_linear_search_params'] = {}
    l_rate_linear_search_params: Param = network_params['learning_rate_linear_search_params']

    l_rate_linear_search_params['max_iterations'] = 1000
    l_rate_linear_search_params['max_value'] = 1
    l_rate_linear_search_params['error_tolerance'] = network_params['error_tolerance']

    # Variable Learning Rate Params
    network_params['variable_learning_rate_params'] = {}
    variable_l_rate_params: Param = network_params['variable_learning_rate_params']

    variable_l_rate_params['down_scaling_factor'] = 0.2
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


def get_training_set(file_name: str, line_count: int, normalize: bool) -> np.ndarray:
    training_set: np.ndarray = pd.read_csv(file_name, delim_whitespace=True, header=None).values
    if normalize:
        training_set = training_set / 100

    if line_count > 1:
        elem_size: int = len(training_set[0]) * line_count
        training_set = np.reshape(training_set, (np.size(training_set) // elem_size, elem_size))

    return training_set

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
        self.network_params['type'] = 'non_linear'
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
        self.network_params['variable_learning_rate_params']['up_scaling_factor'] = \
            self.network_params['variable_learning_rate_params']['down_scaling_factor'] \
            * self.network_params['base_learning_rate']
        neural_network = get_neural_network(self.network_params, len(self.training_points[0]))
        self.min_error = []
        self.last_error = []

        neural_network.train(self.training_points, self.training_values, self.get_network_error)

        self.predictions['linear']['medium_min'] = self.min_error
        self.predictions['linear']['medium_last'] = self.last_error


        # Big
        self.network_params['learning_rate_strategy'] = 'fixed'
        self.network_params['base_learning_rate'] = 0.8
        # self.network_params['variable_learning_rate_params']['up_scaling_factor'] = \
        #     self.network_params['variable_learning_rate_params']['down_scaling_factor'] \
        #     * self.network_params['base_learning_rate']
        neural_network = get_neural_network(self.network_params, len(self.training_points[0]))
        self.min_error = []
        self.last_error = []

        neural_network.train(self.training_points, self.training_values, self.get_network_error)

        self.predictions['linear']['big_min'] = self.min_error
        self.predictions['linear']['big_last'] = self.last_error

        fig = plt.figure(figsize=(16, 10))
        ax = plt.subplot(111)

        ax.plot(self.predictions['linear']['big_last'], color=lighten_color('g', 0.3),  label='big_last')
        ax.plot(self.predictions['linear']['medium_last'], color=lighten_color('m', 0.3), label='medium_last')
        ax.plot(self.predictions['linear']['small_last'], color=lighten_color('k', 0.3), label='small_last')
        ax.plot(self.predictions['linear']['big_min'], 'g-', label='big_min', lw=2)
        ax.plot(self.predictions['linear']['medium_min'], 'm-', label='medium_min', lw=2)
        ax.plot(self.predictions['linear']['small_min'], 'k-', label='small_min', lw=2)
        ax.legend(bbox_to_anchor=(1, 0.7), fancybox=True, shadow=True)

        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.95, box.height])

        ax.semilogy()
        plt.title(f'Error minimo y actual. Momentum: {self.network_params["momentum_factor"]}', pad=20)
        plt.ylabel('Error')
        plt.xlabel('Iteracion')
        plt.show()

        print(self.predictions['linear']['small_min'][-1], self.predictions['linear']['medium_min'][-1], self.predictions['linear']['big_min'][-1])


def main():

    training_points: np.ndarray = get_training_set('trainingset/inputs/Ej2.tsv', 1, True)
    training_values: np.ndarray = get_training_set('trainingset/outputs/Ej2.tsv', 1, True).flatten()

    neural_network: NeuralNetwork = get_neural_network(generate_config(), len(training_points[0]))

    cv = cross_validation(get_neural_network_factory(generate_config(), 3), training_points, training_values, _error, np.size(training_points, axis=0)//20, 1)

    print(cv.best_neural_network.calculate_error(training_points, training_values, insert_identity_column=True))
    print(cv.best_neural_network.calculate_error(cv.best_test_points, cv.best_test_values, insert_identity_column=True))
    print(cv)

if __name__ == "__main__":
    main()