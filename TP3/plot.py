from typing import List, Dict

import matplotlib.pyplot as plt
import numpy as np


def plot_2d_hyperplane(points, values, w) -> None:
    fig = plt.figure()
    fig.tight_layout()
    plt.style.use('ggplot')

    ax = plt.gca()

    positive_values: Dict[str, List[int]] = {
        'x': [],
        'y': []
    }
    negative_values: Dict[str, List[int]] = {
        'x': [],
        'y': []
    }

    for i in range(len(points)):
        if values[i] > 0:
            positive_values['x'].append(points[i][0])
            positive_values['y'].append(points[i][1])
        else:
            negative_values['x'].append(points[i][0])
            negative_values['y'].append(points[i][1])

    x = np.arange(-1.5, 1.5, 0.02)
    y = (- w[1] * x - w[0]) / w[2]

    ax.plot(x, y, 'k')

    ax.scatter(positive_values['x'], positive_values['y'], color='red')
    ax.scatter(negative_values['x'], negative_values['y'], color='blue')

    plt.scatter(positive_values['x'], positive_values['y'], color='red')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Hiperplano 2D')

    plt.show()


def plot_error(error_values: List[float]) -> None:
    fig = plt.figure()
    fig.tight_layout()
    plt.style.use('ggplot')

    ax = plt.gca()
    ax.plot(range(len(error_values)), error_values, 'k')

    ax.set_xlabel('Iteration')
    ax.set_ylabel('Error')
    ax.set_title('Min error at  iteration')

    plt.show()


# Idea
def plot_confusion_matrix(confusion_matrix: np.ndarray):
    fig= plt.figure(figsize=(20, 8))
    fig.set_figwidth(20)
    fig.set_figheight(8)
    fig.tight_layout(pad=3)

    plt.imshow(confusion_matrix, cmap=plt.cm.Blues)
    # plt.set_title(f'Validation {k}')

    # Adds number to heatmap matrix
    for i, j in np.ndindex(confusion_matrix.shape):
        c = confusion_matrix[j][i]
        plt.text(i, j, str(c), va='center', ha='center')
