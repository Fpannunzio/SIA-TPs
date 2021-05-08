import sys

import numpy as np
import pandas
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from TP4.grid import Grid, QuadraticGrid, HexagonalGrid


def exercise(config_file: str):
    europe = pandas.read_csv('europe.csv')
    values = StandardScaler().fit_transform(europe.values[:, 1:])

    pca = PCA()
    pca.fit(values)
    learning_rate = 0.1
    radius = 2
    k = 5

    def distance(point1: np.ndarray, point2: np.ndarray) -> float:
        return np.linalg.norm(point1 - point2)

    grid: HexagonalGrid = HexagonalGrid(learning_rate, radius, k, distance, 7)
    grid.train(len(values)*100, values)
    matrix: np.ndarray = grid.get_near_neurons_mean_distances_matrix()
    print(grid)


if __name__ == "__main__":
    argv = sys.argv

    config_file: str = 'config.yaml'
    if len(argv) > 1:
        config_file = argv[1]

    exercise(config_file)
