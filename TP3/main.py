import math
import sys
from typing import List, Dict, Any, Callable

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


def cross_validation(config_network: Dict[str, Any], training_points: np.ndarray, training_values: np.ndarray, get_metric: Callable[[NeuralNetwork, np.ndarray, np.ndarray], float], size: int, iteration: int) -> [NeuralNetwork, np.ndarray]:

    while len(training_values) % size != 0:
        size += 1

    if size == len(training_values):
        iteration = 1

    gt_points: np.ndarray
    gt_values: np.ndarray
    gv_points: np.ndarray
    gv_values: np.ndarray
    best_indexes: np.ndarray = np.zeros((1, 1))
    best_param: float = 0
    current_param: float = 0
    best_neural_network: NeuralNetwork = get_neural_network(config_network, len(training_points[0]))
    neural_network: NeuralNetwork = get_neural_network(config_network, len(training_points[0]))

    for _ in range(iteration):
        possible_values: np.ndarray = np.arange(len(training_points))

        for i in range(math.floor(np.size(training_values)/size)):
            indexes = np.random.choice(possible_values, size=size, replace=False)
            possible_values = possible_values[~np.isin(possible_values, indexes)]
            gt_points = np.delete(training_points, indexes, axis=0)
            gt_values = np.delete(training_values, indexes, axis=0)
            gv_points = np.take(training_points, indexes, axis=0)
            gv_values = np.take(training_values, indexes, axis=0)

            neural_network.train(gt_points, gt_values)
            current_param = get_metric(neural_network, gv_points, gv_values)
            if best_param < current_param:
                best_param = current_param
                best_indexes = indexes
                best_neural_network = neural_network

    return best_neural_network, best_indexes

def main(config_file: str):
    print(f'Loading config file {config_file}...')
    config: Config = Config(config_file)

    training_set: Param = config.training_set

    training_points: np.ndarray = get_training_set(training_set['inputs'], training_set['input_line_count'], training_set['normalize_values'])

    training_values: np.ndarray = get_training_set(training_set['outputs'], training_set['output_line_count'], training_set['normalize_values'])

    neural_network: NeuralNetwork = get_neural_network(config.network, len(training_points[0]))

    network_error_by_iteration: List[float] = []

    def get_network_error(network: NeuralNetwork, selected_training_point: int) -> None:
        network_error_by_iteration.append(network.error)
        #print(network.l_rate, network.error, network.last_training_error, network.training_iteration)

    def classify(number: float) -> int:
        if number < 0:
            return 0
        return 1

    neural_network.train(training_points, training_values, get_network_error)

    #plot_error(network_error_by_iteration)



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
