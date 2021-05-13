import sys

import numpy as np

from TP4.config import Config, Param
from TP4.config_to_grid import get_grid, get_normalized_values
from TP4.kohonen_grid import KohonenGrid, KohonenQuadraticGrid, KohonenHexagonalGrid


def exercise(config_file: str):
    config: Config = Config(config_file)

    training_set: Param = config.input_set

    values = get_normalized_values(training_set['input'])

    grid: KohonenGrid = get_grid(config.grid, len(values[0]))
    grid.train(len(values)*100, values)
    matrix: np.ndarray = grid.get_near_neurons_mean_distances_matrix()
    print(grid)


if __name__ == "__main__":
    argv = sys.argv

    config_file: str = 'config.yaml'
    if len(argv) > 1:
        config_file = argv[1]

    # try:
    exercise(config_file)

    # except KeyboardInterrupt:
    #     sys.exit(0)
    #
    # except (ValueError, FileNotFoundError) as ex:
    #     print('\nAn Error Was Found!!')
    #     print(ex)
    #     sys.exit(1)
    #
    # except Exception as ex:
    #     print('An unexpected error occurred')
    #     raise ex

