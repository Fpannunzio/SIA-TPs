import math
import sys

import matplotlib.pyplot as plt
import numpy as np

from plot import lighten_color
from config import Param, Config
from config_to_network import get_neural_network_factory, get_training_set
from neural_network_lib.neural_network import NeuralNetwork
from neural_network_lib.neural_network_utils import NeuralNetworkFactory, CrossValidationResult, cross_validation


def ej2(config_file: str):

    PARTITIONS_COUNT: int = 10
    ROUNDS: int = 100

    print(f'Loading config file {config_file}...')
    config: Config = Config(config_file)

    training_set: Param = config.training_set

    training_points: np.ndarray = get_training_set(training_set['inputs'], training_set['input_line_count'], training_set['normalize_values'])

    training_values: np.ndarray = get_training_set(training_set['outputs'], training_set['output_line_count'], training_set['normalize_values'])

    neural_network_factory: NeuralNetworkFactory = get_neural_network_factory(config.network, len(training_points[0]))

    def error_metric(nn: NeuralNetwork, points: np.ndarray, values: np.ndarray) -> float:
        return nn.calculate_error(points, np.squeeze(values), training=False, insert_identity_column=True)

    error_history: np.ndarray = np.zeros((PARTITIONS_COUNT * ROUNDS, config.network['max_training_iterations']))

    def save_errors(network: NeuralNetwork, selected_training_point: int, partition: int) -> None:
        error_history[partition][network.training_iteration - 1] = network.error

    def error_comparator(prev: float, curr: float) -> int:
        return 0 if math.isclose(prev, curr) else np.sign(prev - curr)

    validation_result: CrossValidationResult = cross_validation(
        neural_network_factory, training_points, training_values,
        error_metric, len(training_points)//PARTITIONS_COUNT, ROUNDS,
        metric_comparator=error_comparator, status_callback=save_errors
    )

    print(f'STD: {validation_result.metrics_std}')
    print(f'Mean: {validation_result.metrics_mean}')
    print(f'Best Error: {validation_result.best_metric}')

    if config.plot:

        for error in error_history:
            plt.plot(error, color=lighten_color('g', 0.3))

        l_mean, = plt.plot(np.mean(error_history, axis=0), color='g', lw=3)
        l_best, = plt.plot(error_history[validation_result.best_partition_index], color='k', lw=3)
        plt.xlabel("Iteracion ")
        plt.ylabel("Error")
        plt.title(f'Error en el entrenamiento. Perceptron {config.network["type"]}. \n'
                  f'Error {config.network["network_params"]["error_function"]}. Cantidad de particiones {PARTITIONS_COUNT}.')
        plt.semilogy()
        plt.legend([l_mean, l_best], ['Medio', 'Mejor métrica asociada'])
        plt.show()

        plt.scatter(error_history[:, -1], validation_result.all_metrics)
        plt.xlabel("Error en el entrenamiento")
        plt.ylabel("Metrica del conjunto de validacion (error)")
        plt.title(f'Error vs Metrica. Perceptron {config.network["type"]}. Iteraciones {config.network["max_training_iterations"]}\n'
                  f'Error {config.network["network_params"]["error_function"]}. Cantidad de particiones {PARTITIONS_COUNT}.')
        plt.show()


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
