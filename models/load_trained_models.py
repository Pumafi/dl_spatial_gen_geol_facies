from models.wgan_gs.wgs_generator import WassersteinGSGenerator
from tensorflow import keras


def load_msnwgen_2d_gs_horizontal(checkpoint_file="../trainedweights/msnwgen2d_gs/cp-msnwgen_maxsort_horizontal_good.ckpt",
                                 slice_size = (64, 128, 4),
                                 noise_shape = (8, 16)):
    """
    Load a trained MultiScale Wasserstein Generative Adversarial Network 2D (with Spect. Norm. and GroupSort)
    Args:
        checkpoint_file: placement of the trained weigths checkpoint

    Returns:
        The trained generator model
    """
    g_model = MSNWGSGenerator(output_dims=slice_size)
    g_model.build([None, *noise_shape, 1])
    g_model.load_weights(checkpoint_file)

    return g_model
