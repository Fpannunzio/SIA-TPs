from typing import List, Tuple
import os
import numpy as np
from skimage import color, io
from skimage.transform import resize
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d

from TP3.neural_network_lib.neural_network import MultilayeredNeuralNetwork
from TP5.MLP import Network


def labeled_scatter(xValues, yValues, labels=None):
    plt.figure(figsize=(16, 10))
    xs = xValues
    ys = yValues

    plt.scatter(xs, ys)

    for i in range(len(labels)):
        plt.text(xs[i], ys[i], s=labels[i],
                 fontdict=dict(color='red', size=10))

    plt.xlabel("X")
    plt.xlabel("Y")
    plt.grid()
    plt.show()


def to_bits(values: np.ndarray) -> np.ndarray:
    new_values = np.empty((np.size(values, 0), np.size(values, 1) * 5))

    for i in range(np.size(values, 0)):
        new_values[i] = np.array(
            [[(0.7 if (v >> i & 1) == 1 else -0.7) for i in range(4, -1, -1)] for v in
             values[i]]).flatten()  # dos veces i?

    return new_values


def noisy_copy(point: np.ndarray, noise_factor: float) -> np.ndarray:
    return np.array([point[j] if np.random.uniform(0, 1) < noise_factor else -point[j] for j in range(np.size(point))])


def add_noise(values: np.ndarray, noise_factor: float, noise_copies: int) -> Tuple[np.ndarray, np.ndarray]:
    new_inputs = np.empty((np.size(values, 0) * noise_copies, np.size(values, 1)))
    new_outputs = np.empty((np.size(values, 0) * noise_copies, np.size(values, 1)))

    for i in range(np.size(values, 0)):

        new_inputs[i*noise_copies] = new_outputs[i*noise_copies] = values[i]

        for k in range(1, noise_copies):
            new_index = i*noise_copies + k
            new_inputs[new_index] = noisy_copy(values[i], noise_factor)
            new_outputs[new_index] = values[i]

    return new_inputs, new_outputs


def print_bit_array(bit_array: List[float]):
    number: str = ''
    for i, bit in enumerate(bit_array):
        if i != 0 and i % 5 == 0:
            number += '\n'
        if bit >= 0:
            number += '*'
        else:
            number += ' '
    print(number)


def generate(neural_network: Network, z_values: np.ndarray, fv, sv, n):
    f = interp1d(z_values[(fv, sv), 0], z_values[(fv, sv), 1], kind='linear')

    xnew = np.linspace(z_values[fv, 0], z_values[sv, 0], num=n, endpoint=True)

    z_val = np.empty(2)

    for x in xnew:
        z_val[0] = x
        z_val[1] = f(x)
        print(z_val)
        prediction, w = neural_network.predict_from_layer(z_val, len(neural_network.hidden_layers) // 2 + 1)
        print_bit_array(prediction)

def create_greyscale_dataset_with_repetition(img_folder:str, digit_size:int, multiple_images: int):
    img_data_array = []
    for file in os.listdir(img_folder):
        image_path = img_folder + '/' + file
        image = io.imread(image_path)
        image = color.rgb2gray(image)
        image = resize(image, (digit_size, digit_size), anti_aliasing=True)
        for _ in range(multiple_images):
            img_data_array.append(np.array(image))
    return img_data_array

def add_noise2(images: np.ndarray, noise: float):
    images_with_noise = []
    for image in images:
        images_with_noise.append(np.array([image[i] + np.random.uniform(0.05, 0.1) if image[i] > 0 and np.random.random() < noise else image[i] for i in range(np.size(image))]))

    return np.array(images_with_noise)

