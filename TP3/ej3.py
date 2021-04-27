import math
import sys
from typing import Dict, List

import numpy as np
from matplotlib import pyplot as plt

from TP3.neural_network_lib.neural_network import NeuralNetwork
from neural_network_lib.neural_network_utils import CrossValidationResult, accuracy_metric, cross_validation
from plot import plot_confusion_matrix, lighten_color
from config import Param, Config
from config_to_network import get_neural_network_factory, get_training_set, get_neural_network


# Funcion de ejemplo - No se utiliza
def one_layer(config_file: str):
    min_error: List[float] = []
    last_error: List[float] = []
    predictions: Dict[str, Dict[str, List[float]]] = {'one_layer': {}}

    config: Config = Config(config_file)

    training_set: Param = config.training_set

    # Defaults
    if not training_set or training_set['inputs'] is None:
        training_set['inputs'] = 'trainingset/inputs/Ej3-numbers.tsv'
    if not training_set or training_set['outputs'] is None:
        training_set['outputs'] = 'trainingset/outputs/Ej3-numbers.tsv'
    if not training_set or training_set['normalize_values'] is None:
        training_set['normalize_values'] = False

    def get_network_error(network: NeuralNetwork, selected_training_point: int) -> None:
        min_error.append(network.error)
        last_error.append(network.last_training_error)

    training_points: np.ndarray = get_training_set(training_set['inputs'], training_set['input_line_count'],
                                                   training_set['normalize_values'])

    training_values: np.ndarray = get_training_set(training_set['outputs'], training_set['output_line_count'],
                                                   training_set['normalize_values'])

    # Small
    config.network['learning_rate_strategy'] = 'fixed'
    config.network['base_learning_rate'] = 0.00001
    # self.network_params['variable_learning_rate_params']['up_scaling_factor'] = \
    #     self.network_params['variable_learning_rate_params']['down_scaling_factor'] \
    #     * self.network_params['base_learning_rate']

    config.network['network_params']['error_function'] = 'quadratic'
    config.network['network_params']['activation_function'] = 'tanh'
    config.network['network_params']['activation_slope_factor'] = 0.9
    config.network['network_params']['layer_sizes'] = [10, 1]

    neural_network: NeuralNetwork = get_neural_network(config.network, len(training_points[0]))
    min_error = []
    last_error = []

    neural_network.train(training_points, training_values, get_network_error)

    predictions['one_layer']['small_min'] = min_error
    predictions['one_layer']['small_last'] = last_error

    # Medium
    config.network['learning_rate_strategy'] = 'fixed'
    config.network['base_learning_rate'] = 0.05
    neural_network = get_neural_network(config.network, len(training_points[0]))
    min_error = []
    last_error = []

    neural_network.train(training_points, training_values, get_network_error)

    predictions['one_layer']['medium_min'] = min_error
    predictions['one_layer']['medium_last'] = last_error

    # Big
    config.network['learning_rate_strategy'] = 'fixed'
    config.network['base_learning_rate'] = 0.8
    neural_network = get_neural_network(config.network, len(training_points[0]))
    min_error = []
    last_error = []

    neural_network.train(training_points, training_values, get_network_error)

    predictions['one_layer']['big_min'] = min_error
    predictions['one_layer']['big_last'] = last_error

    # Variable
    config.network['learning_rate_strategy'] = 'variable'
    config.network['base_learning_rate'] = 0.05

    neural_network = get_neural_network(config.network, len(training_points[0]))
    min_error = []
    last_error = []

    neural_network.train(training_points, training_values, get_network_error)

    predictions['one_layer']['variable_min'] = min_error
    predictions['one_layer']['variable_last'] = last_error

    plt.plot(predictions['one_layer']['big_last'], color=lighten_color('g', 0.3), label='big_last')
    plt.plot(predictions['one_layer']['medium_last'], color=lighten_color('m', 0.3), label='medium_last')
    plt.plot(predictions['one_layer']['small_last'], color=lighten_color('k', 0.3), label='small_last')
    plt.plot(predictions['one_layer']['variable_last'], color=lighten_color('c', 0.3), label='variable_last')
    plt.plot(predictions['one_layer']['big_min'], 'g-', label='big_min', lw=2)
    plt.plot(predictions['one_layer']['medium_min'], 'm-', label='medium_min', lw=2)
    plt.plot(predictions['one_layer']['small_min'], 'k-', label='small_min', lw=2)
    plt.plot(predictions['one_layer']['variable_min'], 'c-', label='variable_min', lw=2)
    plt.legend()

    plt.semilogy()
    plt.title('loss per model')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.show()

# Funcion de ejemplo - No se utiliza
def multiple_layer(config_file: str):
    min_error: List[float] = []
    last_error: List[float] = []
    predictions: Dict[str, Dict[str, List[float]]] = {'one_layer': {}}

    config: Config = Config(config_file)

    training_set: Param = config.training_set

    # Defaults
    if not training_set or training_set['inputs'] is None:
        training_set['inputs'] = 'trainingset/inputs/Ej3-numbers.tsv'
    if not training_set or training_set['outputs'] is None:
        training_set['outputs'] = 'trainingset/outputs/Ej3-numbers.tsv'
    if not training_set or training_set['normalize_values'] is None:
        training_set['normalize_values'] = False

    def get_network_error(network: NeuralNetwork, selected_training_point: int) -> None:
        min_error.append(network.error)
        last_error.append(network.last_training_error)

    training_points: np.ndarray = get_training_set(training_set['inputs'], training_set['input_line_count'],
                                                   training_set['normalize_values'])

    training_values: np.ndarray = get_training_set(training_set['outputs'], training_set['output_line_count'],
                                                   training_set['normalize_values'])

    config.network['network_params']['layer_sizes'] = [10, 1]
    predictions['one_layer'] = {}

    # Medium
    config.network['learning_rate_strategy'] = 'fixed'
    config.network['base_learning_rate'] = 0.05

    neural_network = get_neural_network(config.network, len(training_points[0]))
    min_error = []
    last_error = []

    neural_network.train(training_points, training_values, get_network_error)

    predictions['one_layer']['medium_min'] = min_error
    predictions['one_layer']['medium_last'] = last_error

    # Variable
    config.network['learning_rate_strategy'] = 'variable'
    config.network['base_learning_rate'] = 0.05

    neural_network = get_neural_network(config.network, len(training_points[0]))
    min_error = []
    last_error = []

    neural_network.train(training_points, training_values, get_network_error)

    predictions['one_layer']['variable_min'] = min_error
    predictions['one_layer']['variable_last'] = last_error

    config.network['network_params']['layer_sizes'] = [10, 5, 1]
    predictions['multiple_layer_medium'] = {}

    # Medium
    config.network['learning_rate_strategy'] = 'fixed'
    config.network['base_learning_rate'] = 0.5

    neural_network = get_neural_network(config.network, len(training_points[0]))
    min_error = []
    last_error = []

    neural_network.train(training_points, training_values, get_network_error)

    predictions['multiple_layer_medium']['medium_min'] = min_error
    predictions['multiple_layer_medium']['medium_last'] = last_error

    # Variable
    config.network['learning_rate_strategy'] = 'variable'
    config.network['base_learning_rate'] = 0.05

    neural_network = get_neural_network(config.network, len(training_points[0]))
    min_error = []
    last_error = []

    neural_network.train(training_points, training_values, get_network_error)

    predictions['multiple_layer_medium']['variable_min'] = min_error
    predictions['multiple_layer_medium']['variable_last'] = last_error

    config.network['network_params']['layer_sizes'] = [10, 2, 3, 1]
    predictions['multiple_layer_big'] = {}

    # Medium
    config.network['learning_rate_strategy'] = 'fixed'
    config.network['base_learning_rate'] = 0.5

    neural_network = get_neural_network(config.network, len(training_points[0]))
    min_error = []
    last_error = []

    neural_network.train(training_points, training_values, get_network_error)

    predictions['multiple_layer_big']['medium_min'] = min_error
    predictions['multiple_layer_big']['medium_last'] = last_error

    # Variable
    config.network['learning_rate_strategy'] = 'variable'
    config.network['base_learning_rate'] = 0.05

    neural_network = get_neural_network(config.network, len(training_points[0]))
    min_error = []
    last_error = []

    neural_network.train(training_points, training_values, get_network_error)

    predictions['multiple_layer_big']['variable_min'] = min_error
    predictions['multiple_layer_big']['variable_last'] = last_error

    plt.plot(predictions['one_layer']['medium_last'], color=lighten_color('g', 0.3), label='medium [10, 1]')
    plt.plot(predictions['one_layer']['variable_last'], color=lighten_color('c', 0.3), label='variable [10, 1]')
    plt.plot(predictions['multiple_layer_medium']['medium_last'], color=lighten_color('m', 0.3),
             label='medium [10, 5, 1]')
    plt.plot(predictions['multiple_layer_medium']['variable_last'], color=lighten_color('k', 0.3),
             label='variable [10, 5, 1]')
    plt.plot(predictions['multiple_layer_medium']['medium_last'], 'c--',
             label='medium [10, 2, 3, 1]')
    plt.plot(predictions['multiple_layer_medium']['variable_last'], 'g--',
             label='variable [10, 2, 3, 1]')

    plt.legend()

    plt.semilogy()
    plt.title('loss per model - multiple-layers vs one - momentum: 0.5')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.show()


def main(config_file: str):
    PARTITIONS_COUNT: int = 2

    print(f'Loading config file {config_file}...')
    config: Config = Config(config_file)

    training_set: Param = config.training_set

    # Defaults
    if not training_set or training_set['inputs'] is None:
        training_set['inputs'] = 'trainingset/inputs/Ej3-numbers.tsv'
    if not training_set or training_set['outputs'] is None:
        training_set['outputs'] = 'trainingset/outputs/Ej3-numbers.tsv'
    if not training_set or training_set['normalize_values'] is None:
        training_set['normalize_values'] = False

    training_points: np.ndarray = get_training_set(training_set['inputs'], training_set['input_line_count'],
                                                   training_set['normalize_values'])

    training_values: np.ndarray = get_training_set(training_set['outputs'], training_set['output_line_count'],
                                                   training_set['normalize_values'])

    def metric_comparator(prev: float, curr: float) -> int:
        return 0 if math.isclose(prev, curr) else np.sign(curr - prev)

    results: CrossValidationResult = cross_validation(
        get_neural_network_factory(config.network, len(training_points[0])), training_points, training_values,
        lambda nn, tp, tv: accuracy_metric(nn, tp, tv, [0]), len(training_points) // PARTITIONS_COUNT, 10,
        metric_comparator
    )

    print(f'STD: {results.metrics_std}')
    print(f'Mean: {results.metrics_mean}')
    print(f'Best Accuracy: {results.best_metric}')

    if config.plot:
        plot_confusion_matrix(
            results.best_error_network.get_confusion_matrix(
                results.best_error_points, results.best_error_values,
                2,
                lambda x: 1 if x >= 0 else 0,
                True
            ),
            'Confusion matrix for best error network with training points'
        )

        plot_confusion_matrix(
            results.best_neural_network.get_confusion_matrix(
                results.best_training_points, results.best_training_values,
                2,
                lambda x: 1 if x >= 0 else 0,
                True
            ),
            'Confusion matrix for best metric network with training points'
        )

        plot_confusion_matrix(
            results.best_neural_network.get_confusion_matrix(
                training_points,
                training_values,
                2,
                lambda x: 1 if x >= 0 else 0,
                True),
            'Confusion matrix for best metric network with all points'
        )


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
