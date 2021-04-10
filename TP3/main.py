import sys
from typing import Dict

import numpy as np
import pandas as pd

from config import Config
from perceptron import Perceptron, get_perceptron
from plot import AsyncPlotter, get_plotter


def main(config_file: str):
    print(f'Loading config file {config_file}...')
    config: Config = Config(config_file)

    training_set: Dict[str, str] = config.training_set

    training_x: np.ndarray = pd.read_csv(training_set['inputs'], delim_whitespace=True, header=None).values
    training_y: np.ndarray = pd.read_csv(training_set['outputs'], delim_whitespace=True, header=None).values

    perceptron: Perceptron = get_perceptron(config.perceptron)

    plotter: AsyncPlotter = get_plotter(config.plotting, training_x, training_y)

    print(perceptron.generate_hyperplane_coefficients(training_x, training_y, plotter))

    validation_set: Dict[str, str] = config.training_set

    validation_x: np.ndarray = pd.read_csv(validation_set['inputs'], delim_whitespace=True, header=None).values
    validation_y: np.ndarray = pd.read_csv(validation_set['outputs'], delim_whitespace=True, header=None).values

    if perceptron.are_validate_coefficients_valid(validation_x, validation_y):
        print('La solucion encontrada paso la prueba de validacion')
    else:
        print('La solucion encontrada no paso la prueba de validacion')


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
