import numpy as np
import pandas as pd

from TP3.perceptron import Perceptron


def main():
    # x: np.array = np.array([[-1, 1], [1, -1], [-1, -1], [1, 1]])
    # y: np.array = np.array([-1, -1, -1, 1])

    # perceptron: Perceptron = Perceptron(0.01, np.sign)
    # print(perceptron.generate_hyperplane_coefficients(x, y))
    #  XOR es un problema no linealmente separable
    # y: List[int] = [1, 1, -1, -1]

    x: np.ndarray = pd.read_csv('trainingset/trainingset2.txt', delim_whitespace=True, header=None).values
    y: np.ndarray = pd.read_csv('resultset/resultset2.txt', delim_whitespace=True, header=None).values

    perceptron: Perceptron = Perceptron(0.005, lambda var: var)
    print(perceptron.generate_hyperplane_coefficients(x, y))
    # with open('training.txt') as f:
    #     for line in f:
    #         numbers: List[str] = line.split()
    #         x.append([float(numbers[i]) for i in range(len(numbers))])
    #
    #     f.close()
    #
    # with open('result.txt') as f:
    #     for line in f:
    #         y.append(float(line))
    #
    #     f.close()
    #
    # resolve(x, y, lambda var: var, False)
    # resolve(x, y, np.tanh, True)


if __name__ == "__main__":
    main()
