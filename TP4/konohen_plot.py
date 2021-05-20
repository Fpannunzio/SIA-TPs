from matplotlib.cm import get_cmap
from matplotlib.collections import RegularPolyCollection
from matplotlib import cm, colors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


def generate_centers(grid_size: int) -> np.ndarray:
    # vertical_offset: float = 0.8660254
    vertical_offset: float = 1 / (grid_size + 1) / 2
    even_horizontal_offset: float = 1 / (grid_size + 1) / 2
    odd_horizontal_offset: float = 2 / (grid_size + 1) / 2

    ans: np.ndarray = np.zeros((grid_size, grid_size, 2))

    # ans[::2, :, 0] = np.arange(even_horizontal_offset, even_horizontal_offset + grid_size)
    # ans[1::2, :, 0] = np.arange(odd_horizontal_offset, odd_horizontal_offset + grid_size)

    ans[::2, :, 0] = np.linspace(even_horizontal_offset, 1 - odd_horizontal_offset, num=grid_size)
    ans[1::2, :, 0] = np.linspace(odd_horizontal_offset, 1 - even_horizontal_offset, num=grid_size)

    ans = ans.reshape((grid_size ** 2, 2))

    ans[:, 1] = np.repeat(np.linspace(even_horizontal_offset, 1 - odd_horizontal_offset, num=grid_size), grid_size)

    return ans


# https://stackoverflow.com/questions/2334629/hexagonal-self-organizing-map-in-python/23689112#23689112
def plot_map_with_countries(countriesAndHits, g_size: int, colormap: str, title: str):
    """
    Hits: Array con secuencia de neuronas elegidas. En base a eso se cuenta
    cuantas veces cayo en cada una.
    """

    n_centers = generate_centers(g_size)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, aspect='equal')
    ax.set_xlim([0, g_size + 1.5])
    ax.set_ylim([0, g_size + 1.5])
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    width, height = bbox.width * fig.dpi, bbox.height * fig.dpi
    hits = np.histogram(countriesAndHits['hits'].values, bins=g_size ** 2, range=(0, g_size ** 2))[0]
    # Discover difference between centers
    collection = RegularPolyCollection(
        numsides=6,  # a hexagon
        rotation=0, sizes=((height / g_size / 2) ** 2,),
        edgecolors=(0, 0, 0, 1),
        array=hits,
        cmap=getattr(cm, colormap),
        offsets=n_centers,
        transOffset=ax.transAxes,

    )
    ax.axis('off')

    countriesPerNeuron = [None] * g_size ** 2
    for country, neuron in countriesAndHits.values:
        if not countriesPerNeuron[neuron]:
            countriesPerNeuron[neuron] = []
        countriesPerNeuron[neuron].append(country)

    hex_height = 0.8 / g_size

    for i in range(len(countriesPerNeuron)):
        cx = n_centers[i][0]
        cy = n_centers[i][1]

        if not countriesPerNeuron[i]:
            continue

        count = len(countriesPerNeuron[i])

        if count == 1:
            y_positions = [cy]
        else:
            y_positions = np.linspace(cy - hex_height/2, cy + hex_height/2, count)

        for j in range(count):
            ax.text(cx, y_positions[j], countriesPerNeuron[i][j], transform=ax.transAxes,
                    ha='center', fontdict=dict(color='red', size=8),
                    bbox=dict(facecolor='yellow', alpha=0.5))

    ax.set_title(title)

    ax.add_collection(collection, autolim=True)
    fig.colorbar(collection)
    plt.show()

def plot_map(hits, g_size: int, colormap: str, title: str):
    """
    Hits: Array con secuencia de neuronas elegidas. En base a eso se cuenta
    cuantas veces cayo en cada una.
    """

    n_centers = generate_centers(g_size)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, aspect='equal')
    ax.set_xlim([0, g_size + 1.5])
    ax.set_ylim([0, g_size + 1.5])
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    width, height = bbox.width * fig.dpi, bbox.height * fig.dpi
    # Discover difference between centers
    collection = RegularPolyCollection(
        numsides=6,  # a hexagon
        rotation=0, sizes=((height / g_size / 2) ** 2,),
        edgecolors=(0, 0, 0, 1),
        array=hits,
        cmap=getattr(cm, colormap),
        offsets=n_centers,
        transOffset=ax.transAxes,

    )
    ax.axis('off')
    ax.set_title(title)
    ax.add_collection(collection, autolim=True)
    fig.colorbar(collection)
    plt.show()

def plot_max_weight(max_weight_index, features,g_size: int, colormap: str, title: str):
    """
    Hits: Array con secuencia de neuronas elegidas. En base a eso se cuenta
    cuantas veces cayo en cada una.
    """

    n_centers = generate_centers(g_size)

    tab8 = colors.ListedColormap(get_cmap('tab10').colors, N=len(features))

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, aspect='equal')
    ax.set_xlim([0, g_size + 1.5])
    ax.set_ylim([0, g_size + 1.5])
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    width, height = bbox.width * fig.dpi, bbox.height * fig.dpi
    # Discover difference between centers
    collection = RegularPolyCollection(
        numsides=6,  # a hexagon
        rotation=0, sizes=((height / g_size / 2) ** 2,),
        edgecolors=(0, 0, 0, 1),
        array=max_weight_index,
        cmap=tab8,
        offsets=n_centers,
        transOffset=ax.transAxes,

    )
    ax.axis('off')

    legend_elements = [Line2D([0], [0], marker='o', color='w', label=features[i], markerfacecolor=tab8(i), markersize=15) for i in range(len(features))]

    ax.legend(handles=legend_elements, bbox_to_anchor=(1.1, 0.5), fontsize='x-large')
    ax.set_title(title)
    ax.add_collection(collection, autolim=True)
    plt.show()