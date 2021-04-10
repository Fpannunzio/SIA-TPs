import sys
from typing import Dict

import numpy as np
import pandas as pd

from TP3.config import Config
from TP3.perceptron import Perceptron, get_perceptron


def main(config_file: str):
    print(f'Loading config file {config_file}...')
    config: Config = Config(config_file)

    training_set: Dict[str, str] = config.training_set

    training_x: np.ndarray = pd.read_csv(training_set['inputs'], delim_whitespace=True, header=None).values
    training_y: np.ndarray = pd.read_csv(training_set['outputs'], delim_whitespace=True, header=None).values

    perceptron: Perceptron = get_perceptron(config.perceptron)

    print(perceptron.generate_hyperplane_coefficients(training_x, training_y))


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
