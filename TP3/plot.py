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


def plot_confusion_matrix(confusion_matrix: np.ndarray, graph_title: str):
    fig = plt.figure()
    fig.tight_layout()

    plt.imshow(confusion_matrix, cmap=plt.cm.Blues)

    # Adds number to heatmap matrix
    for i, j in np.ndindex(confusion_matrix.shape):
        c = confusion_matrix[j][i]
        plt.text(i, j, str(c), va='center', ha='center')

    plt.title(graph_title)
    plt.ylabel('Guess')
    plt.xlabel('True value')

    plt.show()


# https://stackoverflow.com/a/49601444/12270520
def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])
