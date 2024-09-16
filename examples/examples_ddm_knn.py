import numpy as np
from data.load_data import load_data

from utils.visualisation import get_color_map
import tensorflow as tf
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 13})

cmap, norm = get_color_map(number_of_categories=4)

batch_size = 5

# Loading real data
slice_size = (64, 128)
x = load_data(slice_size[0], slice_size[1], "../data/horizontal/dataFlumyHoriz.csv")


x_ddm_full = np.load("../mjp_ddm_samples_list_saves_1000_T.npy")
for k in range(20):
    x_ddm = x_ddm_full[k*5:(k*5)+batch_size]

    # Initialize a matrix to store cross-entropy losses for each pair
    neighbours = []

    # Loop over each element in set A and set B to compute cross-entropy
    for i, ddm_realisation in enumerate(x_ddm):
        min_loss = 10e5
        neighbour = None
        for j, real_realisation in enumerate(x):

            # Compute pixel-wise cross-entropy between A[i] and B[j]
            loss = tf.keras.losses.CategoricalCrossentropy()(real_realisation, ddm_realisation)
            loss = np.sum(loss)
            if min_loss > loss:
                min_loss = loss
                neighbour = real_realisation
        neighbours.append(neighbour)

    plt.figure(figsize=(7, 15))
    for i in range(batch_size):
        plt.subplot(batch_size*2, 2, i * 2 + 1)

        plt.axis('off')
        plt.imshow(np.argmax(x_ddm[i], axis=-1).reshape((64, 128)),
                   interpolation='nearest', cmap=cmap, norm=norm)

        plt.subplot(batch_size*2, 2, i * 2 + 2)
        plt.axis('off')
        plt.imshow(np.argmax(neighbours[i], axis=-1).reshape((64, 128)),
                   interpolation='nearest', cmap=cmap, norm=norm)
    plt.show()