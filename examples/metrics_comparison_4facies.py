import numpy as np
from data.load_data import load_data
from metrics.evaluate_models import compare_models_morpho
from utils.visualisation import get_color_map
import tensorflow as tf
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 13})

cmap, norm = get_color_map(number_of_categories=4)

batch_size = 240

# Loading real data
slice_size = (64, 128)
x = load_data(slice_size[0], slice_size[1], "../data/horizontal/dataFlumyHoriz.csv")
x_ddm = np.load("../mjp_ddm_samples_4f.npy")[:batch_size]
print(x_ddm.shape)


def mjp_ddm(random_noise):
    return x_ddm


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


models_names = ["NO CHANGE", "FOCAL", "NO GROUP SORT"]

models = [mjp_ddm,]

# Useful constants

cmap, norm = get_color_map(number_of_categories=4)
facies_names = np.array(["Sand, Channel lag", "Sand, Point bar", "Silts, Levee", "Shale, Overbank"])

x_train = tf.cast(x, dtype=tf.float64)

# Plot the morpho comparision
compare_models_morpho(x_train, models, models_names, facies_names, batch_size)
