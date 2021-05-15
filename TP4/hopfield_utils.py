import numpy as np
import pandas as pd


def get_training_set(file_name: str, line_count: int) -> np.ndarray:
    training_set: np.ndarray = pd.read_csv(file_name, delim_whitespace=True, header=None, dtype=int).values

    if line_count > 1:
        elem_size: int = len(training_set[0]) * line_count
        training_set = np.reshape(training_set, (np.size(training_set) // elem_size, elem_size))

    return training_set


def print_letter_and_prediction(letter: np.ndarray, prediction: np.ndarray, row_size: int):
    if np.size(letter) % row_size != 0:
        raise ValueError("La letra no es divisible por el tamaño de fila escogido")

    letter = letter.reshape(row_size, letter.size // row_size)
    prediction = prediction.reshape(row_size, prediction.size // row_size)

    ans: str = ''

    for row in range(letter.size // row_size):
        for i in range(row_size):
            if letter[row][i] == 1:
                ans += '*'
            else:
                ans += ' '

        ans += '\t'

        for i in range(row_size):
            if prediction[row][i] == 1:
                ans += '*'
            else:
                ans += ' '

        ans += '\n'

    print(ans)


def print_letter(letter: np.ndarray, row_size: int):
    if np.size(letter) % row_size != 0:
        raise ValueError("La letra no es divisible por el tamaño de fila escogido")

    for i in range(np.size(letter, 0)):
        if i != 0 and (i % row_size) == 0:
            print('')
        if letter[i] == 1:
            print('*', end='')
        else:
            print(' ', end='')
    print('')


def random_alter(letter: np.ndarray, count: int) -> np.ndarray:
    indexes = np.random.choice(letter.size, count, replace=False)

    ans = np.copy(letter)

    for index in indexes:
        ans[index] *= -1

    return ans
