import sys

import numpy as np
import pandas as pd

from TP4.config import Config, Param
from TP4.hopfield import HopfieldNetwork


def get_training_set(file_name: str, line_count: int) -> np.ndarray:
    training_set: np.ndarray = pd.read_csv(file_name, delim_whitespace=True, header=None, dtype=int).values

    if line_count > 1:
        elem_size: int = len(training_set[0]) * line_count
        training_set = np.reshape(training_set, (np.size(training_set) // elem_size, elem_size))

    return training_set


def print_letter(letter: np.ndarray, row_size: int):
    if np.size(letter) % row_size != 0:
        raise ValueError("La letra no es divisible por el tamaño de fila escogido")

    # print(np.reshape(letter, (int(np.size(letter)/row_size), row_size)))
    for i in range(np.size(letter, 0)):
        if i != 0 and (i % row_size) == 0:
            print('')
        if letter[i] == 1:
            print('*', end='')
        else:
            print(' ', end='')
    print('\n')

def exercise(config_file: str):
    config: Config = Config(config_file)
    rows_per_entry: int = 5
    elem_per_row: int = 5

    training_letters: np.ndarray = get_training_set('patterns/letters_inputs.tsv', rows_per_entry)

    testing_letters: np.ndarray = get_training_set('patterns/letters_test.tsv', rows_per_entry)

    absurd_letter: np.ndarray = get_training_set('patterns/absurd_pattern.tsv', rows_per_entry)

    hopfield_network: HopfieldNetwork = HopfieldNetwork(training_letters)

    for i in range(np.size(training_letters, 0)):
        print_letter(hopfield_network.evaluate(training_letters[i]), elem_per_row)
        print_letter(hopfield_network.evaluate(testing_letters[i]), elem_per_row)

    print_letter(hopfield_network.evaluate(absurd_letter[0]), elem_per_row)

if __name__ == "__main__":
    argv = sys.argv

    config_file: str = 'config.yaml'
    if len(argv) > 1:
        config_file = argv[1]

    # try:
    exercise(config_file)
