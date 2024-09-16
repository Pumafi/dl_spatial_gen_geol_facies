import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from data.load_data import load_data
from metrics.evaluate_models import compare_models_morpho
from utils.visualisation import get_color_map

plt.rcParams.update({'font.size': 13})

cmap, norm = get_color_map(number_of_categories=4)

batch_size = 500

# Loading real data
slice_size = (64, 64)
x = load_data(slice_size[0], slice_size[1], "../data/horizontal/dataFlumyHoriz_9facies.csv", sep=" ")
x_ddm = np.load("../mjp_ddm_samples_9f.npy")[:batch_size]
x_gan = np.load("../gan_samples_list_saves_1000.npy")[:batch_size]
print(x_ddm.shape, x_gan.shape)


def mjp_ddm_9(random_noise):
    return x_ddm

def gan_9(random_noise):
    return x_gan


def show_images(array_img, model_name):
    # IMAGES
    plt.figure(figsize=(10, 5))
    plt.axis('off')
    slice_h = array_img.numpy().shape[1]
    slice_w = array_img.numpy().shape[2]

    size_high_res = (64, 128)

    plt.imshow(np.argmax(array_img.numpy(), axis=-1).reshape(size_high_res),
               interpolation='nearest', cmap=cmap, norm=norm)
    plt.title(model_name)
    plt.show()


models_names = ["MJP DDM", "MSWGAN"]

models = [mjp_ddm_9, gan_9]

# Useful constants

cmap, norm = get_color_map(number_of_categories=4)
facies_names = np.array(["Sand, Channel lag", "Sand, Point bar", "Sand Plug", "Crevasse Splay I",
                         "Crevasse Splay II Channel", "Crevasse Splay", "Silts, Levee", "Shale, Overbank", "Mud Plug"])

x_train = tf.cast(x, dtype=tf.float64)

# Plot the morpho comparision
compare_models_morpho(x_train, models, models_names, facies_names, batch_size)
