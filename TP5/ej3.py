from __future__ import print_function

#!pip install numpy==1.19.5
#!pip install tensorflow==2.2.0

import numpy as np
import matplotlib.pyplot as plt
import skimage.io
from scipy.stats import norm
import tensorflow as tf
from skimage import color, data, io
from skimage.transform import resize

from tensorflow import keras
### hack tf-keras to appear as top level keras
import sys
import os

sys.modules['keras'] = keras
### end of hack

from keras.layers import Input, Dense, Lambda, Reshape
from keras.models import Model
from keras import backend as K
from keras import metrics
from keras.datasets import fashion_mnist

from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()

# defining the key parameters
batch_size = 100

# Parameters of the input images (handwritten digits)
original_dim = 120*120

# Latent space is of dimension 2.  This means that we are reducing the dimension from 784 to 2
latent_dim = 2
intermediate_dim = 1024
epochs = 100
epsilon_std = 1.0
digit_size = 120
noise = 0.45
multiple_images = 20

def create_dataset_with_repetition(img_folder):
    img_data_array = []

        # for dir1 in os.listdir(img_folder):
    for file in os.listdir(img_folder):
        image_path = img_folder + '/' + file
        image = io.imread(image_path)
        image = color.rgb2gray(image)
        image = resize(image, (digit_size, digit_size), anti_aliasing=True)
        for _ in range(multiple_images):
            img_data_array.append(np.array(image))
    return img_data_array

def create_dataset(img_folder):
    img_data_array = []

        # for dir1 in os.listdir(img_folder):
    for file in os.listdir(img_folder):
        image_path = img_folder + '/' + file
        image = io.imread(image_path)
        image = color.rgb2gray(image)
        image = resize(image, (digit_size, digit_size), anti_aliasing=True)
        img_data_array.append(np.array(image))
    return img_data_array

def add_noise(images: np.ndarray):
    images_with_noise = []
    for image in images:
        images_with_noise.append(np.array([image[i] + np.random.uniform(0.05, 0.1) if image[i] > 0 and np.random.random() < noise else image[i] for i in range(np.size(image))]))

    return np.array(images_with_noise)

def sampling(args: tuple):
    # we grab the variables from the tuple
    z_mean, z_log_var = args
    print(z_mean)
    print(z_log_var)
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0.,
                              stddev=epsilon_std)
    return z_mean + K.exp(z_log_var / 2) * epsilon  # h(z)

def ej3():
    # input to our encoder
    x = Input(shape=(original_dim,), name="input")
    # intermediate layer
    h = Dense(intermediate_dim, activation='relu', name="encoding")(x)
    # defining the mean of the latent space
    z_mean = Dense(latent_dim, name="mean")(h)
    # defining the log variance of the latent space
    z_log_var = Dense(latent_dim, name="log-variance")(h)
    # note that "output_shape" isn't necessary with the TensorFlow backend
    z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])
    # defining the encoder as a keras model
    encoder = Model(x, [z_mean, z_log_var, z], name="encoder")
    # print out summary of what we just did
    encoder.summary()

    # Input to the decoder
    input_decoder = Input(shape=(latent_dim,), name="decoder_input")
    # taking the latent space to intermediate dimension
    decoder_h = Dense(intermediate_dim, activation='relu', name="decoder_h")(input_decoder)
    # getting the mean from the original dimension
    x_decoded = Dense(original_dim, activation='sigmoid', name="flat_decoded")(decoder_h)
    # defining the decoder as a keras model
    decoder = Model(input_decoder, x_decoded, name="decoder")
    decoder.summary()

    # grab the output. Recall, that we need to grab the 3rd element our sampling z
    output_combined = decoder(encoder(x)[2])
    # link the input and the overall output
    vae = Model(x, output_combined)
    # print out what the overall model looks like
    vae.summary()

    def vae_loss(x: tf.Tensor, x_decoded_mean: tf.Tensor):
        # Aca se computa la cross entropy entre los "labels" x que son los valores 0/1 de los pixeles, y lo que salió al final del Decoder.
        xent_loss = original_dim * metrics.binary_crossentropy(x, x_decoded_mean) # x-^X
        kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        vae_loss = K.mean(xent_loss + kl_loss)
        return vae_loss

    x_train = y_train = x_test = y_test = np.array(create_dataset_with_repetition('pokemon/selected'))


    plt.imshow(x_train[0], cmap='gray',
               vmin=0, vmax=1)
    plt.show()

    vae.compile(loss=vae_loss, experimental_run_tf_function=False)
    vae.summary()


    # (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    #
    # x_train = x_train.astype('float32') / 255.
    # x_test = x_test.astype('float32') / 255.
    x_train = x_test = add_noise(x_train.reshape((len(x_train), np.prod(x_train.shape[1:]))))



    vae.fit(x_train, x_train,
        shuffle=True,
        epochs=epochs,
        batch_size=batch_size)


    # plt.imshow(decoder.predict(np.array([encoder.predict(x_test, batch_size=batch_size)[0][0]]))[0].reshape(digit_size, digit_size), cmap='gray', vmin=0, vmax=1)
    # plt.show()
    # plt.imshow(decoder.predict(np.array([encoder.predict(x_test, batch_size=batch_size)[0][2]]))[0].reshape(digit_size, digit_size),
    #            cmap='gray', vmin=0, vmax=1)
    # plt.show()
    # plt.imshow(decoder.predict(np.array([encoder.predict(x_test, batch_size=batch_size)[0][3]]))[0].reshape(digit_size, digit_size),
    #            cmap='gray', vmin=0, vmax=1)
    # plt.show()
    # x_test_encoded = encoder.predict(x_test, batch_size=batch_size)[0]
    # plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=y_test, cmap='viridis')
    # plt.colorbar()
    # plt.show()
    n = 5

    figure = np.zeros((digit_size * n, digit_size * n))
    # # linearly spaced coordinates on the unit square were transformed through the inverse CDF (ppf) of the Gaussian
    # # to produce values of the latent variables z, since the prior of the latent space is Gaussian
    grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
    grid_y = norm.ppf(np.linspace(0.05, 0.95, n))
    #
    for i, yi in enumerate(grid_x):
        for j, xi in enumerate(grid_y):
            z_sample = np.array([[xi, yi]])
            x_decoded = decoder.predict(z_sample)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[i * digit_size: (i + 1) * digit_size,
            j * digit_size: (j + 1) * digit_size] = digit

    plt.imshow(figure, cmap='Greys_r')
    plt.show()


if __name__ == "__main__":
    ej3()