import sys
from bisect import bisect
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

from TP3.config import Param, Config
from TP3.config_to_network import get_neural_network, get_neural_network_factory, _accuracy
from TP3.exercises_utils import get_training_set, generate_config, lighten_color
from TP3.neural_network import NeuralNetwork
from TP3.neural_network_utils import cross_validation, CrossValidationResult


def plot_confusion_matrix(confusion_matrix: np.ndarray):
    fig = plt.figure(figsize=(20, 8))
    fig.set_figwidth(20)
    fig.set_figheight(8)
    fig.tight_layout(pad=3)

    plt.imshow(confusion_matrix, cmap=plt.cm.Blues)
    # plt.set_title(f'Validation {k}')

    # Adds number to heatmap matrix
    for i, j in np.ndindex(confusion_matrix.shape):
        c = confusion_matrix[j][i]
        plt.text(i, j, str(c), va='center', ha='center')

    plt.show()

def main(config_file: str):
    print(f'Loading config file {config_file}...')
    config: Config = Config(config_file)

    training_set: Param = config.training_set

    training_points: np.ndarray = get_training_set(training_set['inputs'], training_set['input_line_count'],
                                                   training_set['normalize_values'])

    training_values: np.ndarray = get_training_set(training_set['outputs'], training_set['output_line_count'],
                                                   training_set['normalize_values'])

    neural_network: NeuralNetwork = get_neural_network(config.network, len(training_points[0]))

    results: CrossValidationResult = cross_validation(
        get_neural_network_factory(config.network, len(training_points[0])), training_points, training_values,
        lambda nn, tp, tv: _accuracy(nn, tp, tv, [0]), 4, 15)

    print(f'Best accuracy = {results.best_metric} Mean = {results.metrics_mean} Standard dev= {results.metrics_std}')

    plot_confusion_matrix(
        results.best_error_network.get_confusion_matrix(results.best_error_points, results.best_error_values, 2,
                                                         lambda x: 1 if x >= 0 else 0, True))

    plot_confusion_matrix(results.best_neural_network.get_confusion_matrix(results.best_training_points, results.best_training_values, 2,
                                                                           lambda x: 1 if x >= 0 else 0, True))

    plot_confusion_matrix(results.best_neural_network.get_confusion_matrix(training_points, training_values, 2, lambda x: 1 if x >= 0 else 0, True))


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


class EJ3XOR:

    def __init__(self) -> None:
        self.training_points: np.ndarray = get_training_set('trainingset/inputs/Ej1-XOR.tsv', 1, False)
        self.training_values: np.ndarray = get_training_set('trainingset/outputs/Ej1-XOR.tsv', 1, False)

        self.predictions: Dict[str, Dict[str, List[float]]] = {}
        self.min_error: List[float] = []
        self.last_error: List[float] = []

        self.network_params: Param = generate_config()

    def get_network_error(self, network: NeuralNetwork, selected_training_point: int) -> None:

        self.min_error.append(network.error)
        self.last_error.append(network.last_training_error)

    def one_layer(self):
        self.network_params['type'] = 'multi_layered'
        neural_network: NeuralNetwork

        self.predictions['one_layer'] = {}

        # Small
        self.network_params['learning_rate_strategy'] = 'fixed'
        self.network_params['base_learning_rate'] = 0.00001
        # self.network_params['variable_learning_rate_params']['up_scaling_factor'] = \
        #     self.network_params['variable_learning_rate_params']['down_scaling_factor'] \
        #     * self.network_params['base_learning_rate']

        self.network_params['network_params']['error_function'] = 'quadratic'
        self.network_params['network_params']['activation_function'] = 'tanh'
        self.network_params['network_params']['activation_slope_factor'] = 0.9
        self.network_params['network_params']['layer_sizes'] = [4, 1]

        neural_network = get_neural_network(self.network_params, len(self.training_points[0]))
        self.min_error = []
        self.last_error = []

        neural_network.train(self.training_points, self.training_values, self.get_network_error)

        self.predictions['one_layer']['small_min'] = self.min_error
        self.predictions['one_layer']['small_last'] = self.last_error

        # Medium
        self.network_params['learning_rate_strategy'] = 'fixed'
        self.network_params['base_learning_rate'] = 0.05
        neural_network = get_neural_network(self.network_params, len(self.training_points[0]))
        self.min_error = []
        self.last_error = []

        neural_network.train(self.training_points, self.training_values, self.get_network_error)

        self.predictions['one_layer']['medium_min'] = self.min_error
        self.predictions['one_layer']['medium_last'] = self.last_error

        # Big
        self.network_params['learning_rate_strategy'] = 'fixed'
        self.network_params['base_learning_rate'] = 0.8
        neural_network = get_neural_network(self.network_params, len(self.training_points[0]))
        self.min_error = []
        self.last_error = []

        neural_network.train(self.training_points, self.training_values, self.get_network_error)

        self.predictions['one_layer']['big_min'] = self.min_error
        self.predictions['one_layer']['big_last'] = self.last_error

        # Variable
        self.network_params['learning_rate_strategy'] = 'variable'
        self.network_params['base_learning_rate'] = 0.05

        neural_network = get_neural_network(self.network_params, len(self.training_points[0]))
        self.min_error = []
        self.last_error = []

        neural_network.train(self.training_points, self.training_values, self.get_network_error)

        self.predictions['one_layer']['variable_min'] = self.min_error
        self.predictions['one_layer']['variable_last'] = self.last_error

        plt.plot(self.predictions['one_layer']['big_last'], color=lighten_color('g', 0.3), label='big_last')
        plt.plot(self.predictions['one_layer']['medium_last'], color=lighten_color('m', 0.3), label='medium_last')
        plt.plot(self.predictions['one_layer']['small_last'], color=lighten_color('k', 0.3), label='small_last')
        plt.plot(self.predictions['one_layer']['variable_last'], color=lighten_color('c', 0.3), label='variable_last')
        plt.plot(self.predictions['one_layer']['big_min'], 'g-', label='big_min', lw=2)
        plt.plot(self.predictions['one_layer']['medium_min'], 'm-', label='medium_min', lw=2)
        plt.plot(self.predictions['one_layer']['small_min'], 'k-', label='small_min', lw=2)
        plt.plot(self.predictions['one_layer']['variable_min'], 'c-', label='variable_min', lw=2)
        plt.legend()

        plt.semilogy()
        plt.title('loss per model - with variable - momentum: 0.1')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.show()

    def multiple_layer(self):
        self.network_params['type'] = 'multi_layered'
        neural_network: NeuralNetwork

        self.predictions['one_layer'] = {}

        self.network_params['network_params']['error_function'] = 'quadratic'
        self.network_params['network_params']['activation_function'] = 'tanh'
        self.network_params['network_params']['activation_slope_factor'] = 0.9

        # Big
        self.network_params['learning_rate_strategy'] = 'fixed'
        self.network_params['base_learning_rate'] = 0.5

        self.network_params['network_params']['layer_sizes'] = [4, 1]
        neural_network = get_neural_network(self.network_params, len(self.training_points[0]))
        self.min_error = []
        self.last_error = []

        neural_network.train(self.training_points, self.training_values, self.get_network_error)

        self.predictions['one_layer']['big_min'] = self.min_error
        self.predictions['one_layer']['big_last'] = self.last_error

        # Variable
        self.network_params['learning_rate_strategy'] = 'variable'
        self.network_params['base_learning_rate'] = 0.05

        neural_network = get_neural_network(self.network_params, len(self.training_points[0]))
        self.min_error = []
        self.last_error = []

        neural_network.train(self.training_points, self.training_values, self.get_network_error)

        self.predictions['one_layer']['variable_min'] = self.min_error
        self.predictions['one_layer']['variable_last'] = self.last_error

        self.network_params['type'] = 'multi_layered'
        neural_network: NeuralNetwork

        self.predictions['multiple_layer'] = {}

        self.network_params['network_params']['error_function'] = 'quadratic'
        self.network_params['network_params']['activation_function'] = 'tanh'
        self.network_params['network_params']['activation_slope_factor'] = 0.9

        # Big
        self.network_params['learning_rate_strategy'] = 'fixed'
        self.network_params['base_learning_rate'] = 0.5

        self.network_params['network_params']['layer_sizes'] = [4, 2, 6, 1]
        neural_network = get_neural_network(self.network_params, len(self.training_points[0]))
        self.min_error = []
        self.last_error = []

        neural_network.train(self.training_points, self.training_values, self.get_network_error)

        self.predictions['multiple_layer']['big_min'] = self.min_error
        self.predictions['multiple_layer']['big_last'] = self.last_error

        # Variable
        self.network_params['learning_rate_strategy'] = 'variable'
        self.network_params['base_learning_rate'] = 0.05

        neural_network = get_neural_network(self.network_params, len(self.training_points[0]))
        self.min_error = []
        self.last_error = []

        neural_network.train(self.training_points, self.training_values, self.get_network_error)

        self.predictions['multiple_layer']['variable_min'] = self.min_error
        self.predictions['multiple_layer']['variable_last'] = self.last_error

        plt.plot(self.predictions['one_layer']['big_min'], 'g-', label='big_min_one', lw=2)
        plt.plot(self.predictions['one_layer']['variable_min'], 'c-', label='variable_min_one', lw=2)
        plt.plot(self.predictions['multiple_layer']['big_min'], 'm-', label='big_min_multiple', lw=2)
        plt.plot(self.predictions['multiple_layer']['variable_min'], 'k-', label='variable_min_multiple', lw=2)
        plt.plot(self.predictions['one_layer']['big_last'], 'k--', label='big_last_one')
        plt.plot(self.predictions['one_layer']['variable_last'], 'k--', label='variable_last_one')
        plt.plot(self.predictions['multiple_layer']['big_last'], 'k--', label='big_last_multiple')
        plt.plot(self.predictions['multiple_layer']['variable_last'], 'k--', label='variable_last_multiple')

        plt.legend()

        plt.semilogy()
        plt.title('loss per model - multiple-layers vs one - momentum: 0.1')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.show()

class EJ3EVEN:

    def __init__(self) -> None:
        self.training_points: np.ndarray = get_training_set('trainingset/inputs/Ej3-numbers.tsv', 7, False)
        self.training_values: np.ndarray = get_training_set('trainingset/outputs/Ej3-numbers.tsv', 1, False)

        self.predictions: Dict[str, Dict[str, List[float]]] = {}
        self.min_error: List[float] = []
        self.last_error: List[float] = []

        self.network_params: Param = generate_config()

    def get_network_error(self, network: NeuralNetwork, selected_training_point: int) -> None:
        self.min_error.append(network.error)
        self.last_error.append(network.last_training_error)

    def one_layer(self):
        self.network_params['type'] = 'multi_layered'
        neural_network: NeuralNetwork

        self.predictions['one_layer'] = {}

        # Small
        self.network_params['learning_rate_strategy'] = 'fixed'
        self.network_params['base_learning_rate'] = 0.00001
        # self.network_params['variable_learning_rate_params']['up_scaling_factor'] = \
        #     self.network_params['variable_learning_rate_params']['down_scaling_factor'] \
        #     * self.network_params['base_learning_rate']

        self.network_params['network_params']['error_function'] = 'quadratic'
        self.network_params['network_params']['activation_function'] = 'tanh'
        self.network_params['network_params']['activation_slope_factor'] = 0.9
        self.network_params['network_params']['layer_sizes'] = [10, 1]

        neural_network = get_neural_network(self.network_params, len(self.training_points[0]))
        self.min_error = []
        self.last_error = []

        neural_network.train(self.training_points, self.training_values, self.get_network_error)

        self.predictions['one_layer']['small_min'] = self.min_error
        self.predictions['one_layer']['small_last'] = self.last_error

        # Medium
        self.network_params['learning_rate_strategy'] = 'fixed'
        self.network_params['base_learning_rate'] = 0.05
        neural_network = get_neural_network(self.network_params, len(self.training_points[0]))
        self.min_error = []
        self.last_error = []

        neural_network.train(self.training_points, self.training_values, self.get_network_error)

        self.predictions['one_layer']['medium_min'] = self.min_error
        self.predictions['one_layer']['medium_last'] = self.last_error

        # Big
        self.network_params['learning_rate_strategy'] = 'fixed'
        self.network_params['base_learning_rate'] = 0.8
        neural_network = get_neural_network(self.network_params, len(self.training_points[0]))
        self.min_error = []
        self.last_error = []

        neural_network.train(self.training_points, self.training_values, self.get_network_error)

        self.predictions['one_layer']['big_min'] = self.min_error
        self.predictions['one_layer']['big_last'] = self.last_error

        # Variable
        self.network_params['learning_rate_strategy'] = 'variable'
        self.network_params['base_learning_rate'] = 0.05

        neural_network = get_neural_network(self.network_params, len(self.training_points[0]))
        self.min_error = []
        self.last_error = []

        neural_network.train(self.training_points, self.training_values, self.get_network_error)

        self.predictions['one_layer']['variable_min'] = self.min_error
        self.predictions['one_layer']['variable_last'] = self.last_error

        plt.plot(self.predictions['one_layer']['big_last'], color=lighten_color('g', 0.3), label='big: 0.8')
        plt.plot(self.predictions['one_layer']['medium_last'], color=lighten_color('m', 0.3), label='medium: 0.05')
        plt.plot(self.predictions['one_layer']['small_last'], color=lighten_color('k', 0.3), label='small: 0.005')
        plt.plot(self.predictions['one_layer']['variable_last'], color=lighten_color('c', 0.3), label='variable')
        # plt.plot(self.predictions['one_layer']['big_min'], 'g--', label='big_min', lw=2)
        # plt.plot(self.predictions['one_layer']['medium_min'], 'm--', label='medium_min', lw=2)
        # plt.plot(self.predictions['one_layer']['small_min'], 'k--', label='small_min', lw=2)
        # plt.plot(self.predictions['one_layer']['variable_min'], 'c--', label='variable_min', lw=2)
        plt.legend()

        plt.semilogy()
        plt.title('loss per model - with variable - momentum: 0.1')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.show()

    def multiple_layer(self):
        self.network_params['type'] = 'multi_layered'
        neural_network: NeuralNetwork
        self.network_params['network_params']['error_function'] = 'quadratic'
        self.network_params['network_params']['activation_function'] = 'tanh'
        self.network_params['network_params']['activation_slope_factor'] = 0.9

        self.network_params['network_params']['layer_sizes'] = [10, 1]
        self.predictions['one_layer'] = {}

        # Medium
        self.network_params['learning_rate_strategy'] = 'fixed'
        self.network_params['base_learning_rate'] = 0.05

        neural_network = get_neural_network(self.network_params, len(self.training_points[0]))
        self.min_error = []
        self.last_error = []

        neural_network.train(self.training_points, self.training_values, self.get_network_error)

        self.predictions['one_layer']['medium_min'] = self.min_error
        self.predictions['one_layer']['medium_last'] = self.last_error

        # Variable
        self.network_params['learning_rate_strategy'] = 'variable'
        self.network_params['base_learning_rate'] = 0.05

        neural_network = get_neural_network(self.network_params, len(self.training_points[0]))
        self.min_error = []
        self.last_error = []

        neural_network.train(self.training_points, self.training_values, self.get_network_error)

        self.predictions['one_layer']['variable_min'] = self.min_error
        self.predictions['one_layer']['variable_last'] = self.last_error

        self.network_params['network_params']['layer_sizes'] = [10, 5, 1]
        self.predictions['multiple_layer_medium'] = {}

        # Medium
        self.network_params['learning_rate_strategy'] = 'fixed'
        self.network_params['base_learning_rate'] = 0.5

        neural_network = get_neural_network(self.network_params, len(self.training_points[0]))
        self.min_error = []
        self.last_error = []

        neural_network.train(self.training_points, self.training_values, self.get_network_error)

        self.predictions['multiple_layer_medium']['medium_min'] = self.min_error
        self.predictions['multiple_layer_medium']['medium_last'] = self.last_error

        # Variable
        self.network_params['learning_rate_strategy'] = 'variable'
        self.network_params['base_learning_rate'] = 0.05

        neural_network = get_neural_network(self.network_params, len(self.training_points[0]))
        self.min_error = []
        self.last_error = []

        neural_network.train(self.training_points, self.training_values, self.get_network_error)

        self.predictions['multiple_layer_medium']['variable_min'] = self.min_error
        self.predictions['multiple_layer_medium']['variable_last'] = self.last_error

        self.network_params['network_params']['layer_sizes'] = [10, 2, 3, 1]
        self.predictions['multiple_layer_big'] = {}

        # Medium
        self.network_params['learning_rate_strategy'] = 'fixed'
        self.network_params['base_learning_rate'] = 0.5

        neural_network = get_neural_network(self.network_params, len(self.training_points[0]))
        self.min_error = []
        self.last_error = []

        neural_network.train(self.training_points, self.training_values, self.get_network_error)

        self.predictions['multiple_layer_big']['medium_min'] = self.min_error
        self.predictions['multiple_layer_big']['medium_last'] = self.last_error

        # Variable
        self.network_params['learning_rate_strategy'] = 'variable'
        self.network_params['base_learning_rate'] = 0.05

        neural_network = get_neural_network(self.network_params, len(self.training_points[0]))
        self.min_error = []
        self.last_error = []

        neural_network.train(self.training_points, self.training_values, self.get_network_error)

        self.predictions['multiple_layer_big']['variable_min'] = self.min_error
        self.predictions['multiple_layer_big']['variable_last'] = self.last_error



        plt.plot(self.predictions['one_layer']['medium_last'], color=lighten_color('g', 0.3), label='medium [10, 1]')
        plt.plot(self.predictions['one_layer']['variable_last'], color=lighten_color('c', 0.3), label='variable [10, 1]')
        plt.plot(self.predictions['multiple_layer_medium']['medium_last'], color=lighten_color('m', 0.3), label='medium [10, 5, 1]')
        plt.plot(self.predictions['multiple_layer_medium']['variable_last'], color=lighten_color('k', 0.3), label='variable [10, 5, 1]')
        plt.plot(self.predictions['multiple_layer_medium']['medium_last'], 'c--',
                 label='medium [10, 2, 3, 1]')
        plt.plot(self.predictions['multiple_layer_medium']['variable_last'], 'g--',
                 label='variable [10, 2, 3, 1]')
        # plt.plot(self.predictions['one_layer']['big_min'], 'g-', label='big_min_one', lw=2)
        # plt.plot(self.predictions['one_layer']['variable_min'], 'c-', label='variable_min_one', lw=2)
        # plt.plot(self.predictions['multiple_layer']['big_min'], 'm-', label='big_min_multiple', lw=2)
        # plt.plot(self.predictions['multiple_layer']['variable_min'], 'k-', label='variable_min_multiple', lw=2)

        plt.legend()

        plt.semilogy()
        plt.title('loss per model - multiple-layers vs one - momentum: 0.5')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.show()
